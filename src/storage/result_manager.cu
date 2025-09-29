#include "result_manager.h"
#include "../utils/logger.h"
#include "../utils/format_utils.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sqlite3.h>
#include <openssl/evp.h>

namespace bitcoin_miner {

class ResultManager::Impl {
public:
    Impl() : auto_save(true), max_memory_results(1000), result_count(0) {}
    
    ~Impl() {
        cleanup();
    }
    
    bool initialize(const std::string& output_dir) {
        try {
            this->output_dir = output_dir;
            
            // ایجاد دایرکتوری خروجی
            std::string command = "mkdir -p " + output_dir;
            system(command.c_str());
            
            // راه‌اندازی پایگاه داده
            if (!initialize_database()) {
                return false;
            }
            
            // راه‌اندازی سیستم فایل
            results_file = output_dir + "/mining_results.dat";
            backup_file = output_dir + "/backup_results.dat";
            
            Logger::info("Result manager initialized: {}", output_dir);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Result manager initialization failed: {}", e.what());
            return false;
        }
    }
    
    void set_auto_save(bool enabled) {
        auto_save = enabled;
    }
    
    void set_max_memory_usage(size_t max_mb) {
        max_memory_results = (max_mb * 1024 * 1024) / sizeof(MiningResult);
    }
    
    void save_result(const MiningResult& result) {
        std::lock_guard<std::mutex> lock(mutex);
        
        try {
            // اعتبارسنجی نتیجه
            if (!result.validate()) {
                Logger::warning("Invalid result skipped: {}", result.address);
                return;
            }
            
            // افزودن به حافظه
            memory_results.push_back(result);
            result_count++;
            
            // افزودن به پایگاه داده
            save_to_database(result);
            
            // ذخیره‌سازی خودکار اگر فعال باشد
            if (auto_save && result_count % 100 == 0) {
                save_to_file();
            }
            
            // مدیریت حافظه
            if (memory_results.size() > max_memory_results) {
                memory_results.erase(memory_results.begin());
            }
            
            Logger::debug("Result saved: {}", result.address);
            
        } catch (const std::exception& e) {
            Logger::error("Failed to save result: {}", e.what());
        }
    }
    
    void save_results_batch(const std::vector<MiningResult>& results) {
        std::lock_guard<std::mutex> lock(mutex);
        
        try {
            for (const auto& result : results) {
                if (result.validate()) {
                    memory_results.push_back(result);
                    result_count++;
                    save_to_database(result);
                }
            }
            
            if (auto_save) {
                save_to_file();
            }
            
            Logger::info("Saved {} results in batch", results.size());
            
        } catch (const std::exception& e) {
            Logger::error("Failed to save results batch: {}", e.what());
        }
    }
    
    std::vector<MiningResult> load_results(const std::string& filter) const {
        std::lock_guard<std::mutex> lock(mutex);
        
        std::vector<MiningResult> results;
        
        try {
            if (filter.empty()) {
                return memory_results;
            }
            
            // فیلتر کردن نتایج
            for (const auto& result : memory_results) {
                if (result.address.find(filter) != std::string::npos ||
                    result.private_key_hex.find(filter) != std::string::npos) {
                    results.push_back(result);
                }
            }
            
        } catch (const std::exception& e) {
            Logger::error("Failed to load results: {}", e.what());
        }
        
        return results;
    }
    
    bool contains_result(const std::string& address) const {
        std::lock_guard<std::mutex> lock(mutex);
        
        for (const auto& result : memory_results) {
            if (result.address == address) {
                return true;
            }
        }
        
        return false;
    }
    
    size_t get_result_count() const {
        return result_count;
    }
    
    size_t get_database_size() const {
        // محاسبه اندازه پایگاه داده
        std::ifstream file(results_file, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            return file.tellg();
        }
        return 0;
    }
    
    void print_statistics() const {
        Logger::info("Result Manager Statistics:");
        Logger::info("  Total results: {}", result_count);
        Logger::info("  Memory usage: {} results", memory_results.size());
        Logger::info("  Database size: {} bytes", get_database_size());
        Logger::info("  Auto-save: {}", auto_save ? "enabled" : "disabled");
    }
    
    bool export_to_csv(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(mutex);
        
        try {
            std::ofstream file(filename);
            if (!file.is_open()) {
                Logger::error("Cannot create CSV file: {}", filename);
                return false;
            }
            
            // هدر CSV
            file << "Timestamp,Address,PrivateKeyHex,PrivateKeyWIF,PublicKey,AddressType,Valid\n";
            
            // داده‌ها
            for (const auto& result : memory_results) {
                file << result.timestamp << ","
                     << result.address << ","
                     << result.private_key_hex << ","
                     << result.private_key_wif << ","
                     << result.public_key_compressed_hex << ","
                     << result.address_type << ","
                     << (result.is_valid ? "true" : "false") << "\n";
            }
            
            file.close();
            Logger::info("Exported {} results to CSV: {}", memory_results.size(), filename);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to export to CSV: {}", e.what());
            return false;
        }
    }
    
    bool export_to_json(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(mutex);
        
        try {
            std::ofstream file(filename);
            if (!file.is_open()) {
                Logger::error("Cannot create JSON file: {}", filename);
                return false;
            }
            
            file << "{\n";
            file << "  \"results\": [\n";
            
            for (size_t i = 0; i < memory_results.size(); ++i) {
                const auto& result = memory_results[i];
                file << "    " << result.to_json();
                
                if (i < memory_results.size() - 1) {
                    file << ",";
                }
                file << "\n";
            }
            
            file << "  ],\n";
            file << "  \"metadata\": {\n";
            file << "    \"total_results\": " << result_count << ",\n";
            file << "    \"export_timestamp\": " << std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() << ",\n";
            file << "    \"version\": \"2.0.0\"\n";
            file << "  }\n";
            file << "}\n";
            
            file.close();
            Logger::info("Exported {} results to JSON: {}", memory_results.size(), filename);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to export to JSON: {}", e.what());
            return false;
        }
    }
    
    bool backup_database(const std::string& backup_path) const {
        std::lock_guard<std::mutex> lock(mutex);
        
        try {
            // کپی فایل اصلی به پشتیبان
            std::ifstream src(results_file, std::ios::binary);
            std::ofstream dst(backup_path, std::ios::binary);
            
            if (!src.is_open() || !dst.is_open()) {
                Logger::error("Cannot create backup: {} -> {}", results_file, backup_path);
                return false;
            }
            
            dst << src.rdbuf();
            
            src.close();
            dst.close();
            
            Logger::info("Database backed up to: {}", backup_path);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to backup database: {}", e.what());
            return false;
        }
    }

private:
    std::string output_dir;
    std::string results_file;
    std::string backup_file;
    std::vector<MiningResult> memory_results;
    std::atomic<size_t> result_count;
    std::atomic<bool> auto_save;
    size_t max_memory_results;
    mutable std::mutex mutex;
    sqlite3* database{nullptr};
    
    bool initialize_database() {
        std::string db_file = output_dir + "/results.db";
        
        int rc = sqlite3_open(db_file.c_str(), &database);
        if (rc != SQLITE_OK) {
            Logger::error("Cannot open database: {}", sqlite3_errmsg(database));
            return false;
        }
        
        // ایجاد جدول
        const char* create_table_sql = 
            "CREATE TABLE IF NOT EXISTS mining_results ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "timestamp INTEGER NOT NULL,"
            "address TEXT UNIQUE NOT NULL,"
            "private_key_hex TEXT NOT NULL,"
            "private_key_wif TEXT NOT NULL,"
            "public_key_compressed TEXT NOT NULL,"
            "address_type TEXT NOT NULL,"
            "is_valid BOOLEAN NOT NULL,"
            "verified BOOLEAN NOT NULL DEFAULT 0,"
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP"
            ");";
        
        rc = sqlite3_exec(database, create_table_sql, nullptr, nullptr, nullptr);
        if (rc != SQLITE_OK) {
            Logger::error("Cannot create table: {}", sqlite3_errmsg(database));
            sqlite3_close(database);
            database = nullptr;
            return false;
        }
        
        Logger::info("Database initialized: {}", db_file);
        return true;
    }
    
    void save_to_database(const MiningResult& result) {
        if (!database) return;
        
        sqlite3_stmt* stmt;
        const char* insert_sql = 
            "INSERT OR REPLACE INTO mining_results "
            "(timestamp, address, private_key_hex, private_key_wif, public_key_compressed, address_type, is_valid) "
            "VALUES (?, ?, ?, ?, ?, ?, ?);";
        
        int rc = sqlite3_prepare_v2(database, insert_sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            Logger::error("Failed to prepare SQL statement: {}", sqlite3_errmsg(database));
            return;
        }
        
        sqlite3_bind_int64(stmt, 1, result.timestamp);
        sqlite3_bind_text(stmt, 2, result.address.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, result.private_key_hex.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, result.private_key_wif.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 5, result.public_key_compressed_hex.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 6, result.address_type.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 7, result.is_valid ? 1 : 0);
        
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            Logger::error("Failed to insert result: {}", sqlite3_errmsg(database));
        }
        
        sqlite3_finalize(stmt);
    }
    
    void save_to_file() {
        try {
            std::ofstream file(results_file, std::ios::binary);
            if (!file.is_open()) {
                Logger::error("Cannot open results file: {}", results_file);
                return;
            }
            
            // ذخیره‌سازی باینری
            for (const auto& result : memory_results) {
                file.write(reinterpret_cast<const char*>(&result), sizeof(MiningResult));
            }
            
            file.close();
            
        } catch (const std::exception& e) {
            Logger::error("Failed to save results to file: {}", e.what());
        }
    }
    
    void cleanup() {
        if (database) {
            sqlite3_close(database);
            database = nullptr;
        }
        
        if (auto_save) {
            save_to_file();
        }
    }
};

// پیاده‌سازی متدهای اصلی ResultManager
ResultManager::ResultManager() : impl(new Impl()) {}
ResultManager::~ResultManager() = default;

bool ResultManager::initialize(const std::string& output_dir) {
    return impl->initialize(output_dir);
}

void ResultManager::set_auto_save(bool enabled) {
    impl->set_auto_save(enabled);
}

void ResultManager::set_max_memory_usage(size_t max_mb) {
    impl->set_max_memory_usage(max_mb);
}

void ResultManager::save_result(const MiningResult& result) {
    impl->save_result(result);
}

void ResultManager::save_results_batch(const std::vector<MiningResult>& results) {
    impl->save_results_batch(results);
}

std::vector<MiningResult> ResultManager::load_results(const std::string& filter) const {
    return impl->load_results(filter);
}

bool ResultManager::contains_result(const std::string& address) const {
    return impl->contains_result(address);
}

size_t ResultManager::get_result_count() const {
    return impl->get_result_count();
}

size_t ResultManager::get_database_size() const {
    return impl->get_database_size();
}

void ResultManager::print_statistics() const {
    impl->print_statistics();
}

bool ResultManager::export_to_csv(const std::string& filename) const {
    return impl->export_to_csv(filename);
}

bool ResultManager::export_to_json(const std::string& filename) const {
    return impl->export_to_json(filename);
}

bool ResultManager::backup_database(const std::string& backup_path) const {
    return impl->backup_database(backup_path);
}

} // namespace bitcoin_miner