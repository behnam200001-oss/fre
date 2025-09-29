#include "config_manager.h"
#include "../utils/logger.h"
#include "../utils/format_utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace bitcoin_miner {

class ConfigManager::Impl {
public:
    Impl() {
        set_defaults();
    }
    
    ~Impl() = default;
    
    bool load_from_file(const std::string& filename) {
        try {
            std::ifstream file(filename);
            if (!file.is_open()) {
                Logger::error("Cannot open config file: {}", filename);
                return false;
            }
            
            std::string line;
            std::string current_section;
            
            while (std::getline(file, line)) {
                // حذف فضاهای خاطر و کامنت‌ها
                line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
                if (line.empty() || line[0] == '#' || line[0] == ';') {
                    continue;
                }
                
                // بخش جدید
                if (line[0] == '[' && line.back() == ']') {
                    current_section = line.substr(1, line.length() - 2);
                    continue;
                }
                
                // کلید-مقدار
                size_t delimiter_pos = line.find('=');
                if (delimiter_pos != std::string::npos) {
                    std::string key = line.substr(0, delimiter_pos);
                    std::string value = line.substr(delimiter_pos + 1);
                    
                    std::string full_key = current_section + "." + key;
                    config_data[full_key] = value;
                }
            }
            
            file.close();
            
            // اعتبارسنجی پیکربندی
            if (!validate()) {
                Logger::error("Config validation failed");
                return false;
            }
            
            Logger::info("Config loaded from: {}", filename);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to load config: {}", e.what());
            return false;
        }
    }
    
    bool save_to_file(const std::string& filename) const {
        try {
            std::ofstream file(filename);
            if (!file.is_open()) {
                Logger::error("Cannot create config file: {}", filename);
                return false;
            }
            
            file << "# Advanced Bitcoin Miner CUDA - Configuration File\n";
            file << "# Version: 2.0.0\n\n";
            
            // گروه‌بندی کلیدها بر اساس بخش
            std::map<std::string, std::vector<std::pair<std::string, std::string>>> sections;
            
            for (const auto& [key, value] : config_data) {
                size_t dot_pos = key.find('.');
                if (dot_pos != std::string::npos) {
                    std::string section = key.substr(0, dot_pos);
                    std::string subkey = key.substr(dot_pos + 1);
                    sections[section].emplace_back(subkey, value);
                }
            }
            
            // نوشتن بخش‌ها
            for (const auto& [section, keys] : sections) {
                file << "[" << section << "]\n";
                for (const auto& [key, value] : keys) {
                    file << key << " = " << value << "\n";
                }
                file << "\n";
            }
            
            file.close();
            Logger::info("Config saved to: {}", filename);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to save config: {}", e.what());
            return false;
        }
    }
    
    bool load_from_string(const std::string& config_data_str) {
        // پیاده‌سازی ساده - می‌تواند بهبود یابد
        std::stringstream ss(config_data_str);
        std::string line;
        
        while (std::getline(ss, line)) {
            size_t delimiter_pos = line.find('=');
            if (delimiter_pos != std::string::npos) {
                std::string key = line.substr(0, delimiter_pos);
                std::string value = line.substr(delimiter_pos + 1);
                config_data[key] = value;
            }
        }
        
        return validate();
    }
    
    GPUConfig get_gpu_config(int device_id) const {
        GPUConfig config;
        config.device_id = device_id;
        config.enabled = get_bool("gpu.enabled_devices." + std::to_string(device_id), true);
        config.workload_percentage = get_int("gpu.workload_distribution." + std::to_string(device_id), 25);
        config.batch_size = get_int("gpu.batch_size", 1000000);
        config.threads_per_block = get_int("gpu.threads_per_block", 256);
        config.max_utilization = get_int("gpu.max_utilization", 85);
        return config;
    }
    
    BloomFilterConfig get_bloom_config() const {
        BloomFilterConfig config;
        config.expected_elements = get_int("bloom_filter.expected_elements", 50000000);
        config.false_positive_rate = get_double("bloom_filter.false_positive_rate", 0.000001); // 1e-6
        config.memory_limit_mb = get_int("bloom_filter.memory_limit_mb", 200);
        config.gpu_accelerated = get_bool("bloom_filter.gpu_accelerated", true);
        return config;
    }
    
    MiningConfig get_mining_config() const {
        MiningConfig config;
        config.keys_per_batch = get_int("mining.keys_per_batch", 1000000);
        config.max_concurrent_batches = get_int("mining.max_concurrent_batches", 8);
        config.compressed_addresses = get_bool("mining.compressed_addresses", true);
        config.enable_all_formats = get_bool("mining.enable_all_formats", true);
        config.address_version = static_cast<uint8_t>(get_int("mining.address_version", 0x00));
        config.testnet_mode = get_bool("mining.testnet_mode", false);
        return config;
    }
    
    std::string get_string(const std::string& key, const std::string& default_val) const {
        auto it = config_data.find(key);
        if (it != config_data.end()) {
            return it->second;
        }
        return default_val;
    }
    
    int get_int(const std::string& key, int default_val) const {
        auto it = config_data.find(key);
        if (it != config_data.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                Logger::warning("Invalid integer value for key: {}", key);
            }
        }
        return default_val;
    }
    
    double get_double(const std::string& key, double default_val) const {
        auto it = config_data.find(key);
        if (it != config_data.end()) {
            try {
                return std::stod(it->second);
            } catch (...) {
                Logger::warning("Invalid double value for key: {}", key);
            }
        }
        return default_val;
    }
    
    bool get_bool(const std::string& key, bool default_val) const {
        auto it = config_data.find(key);
        if (it != config_data.end()) {
            std::string value = it->second;
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            return (value == "true" || value == "1" || value == "yes" || value == "on");
        }
        return default_val;
    }
    
    bool validate() const {
        std::vector<std::string> errors;
        
        // اعتبارسنجی پارامترهای Bloom Filter
        auto bloom_config = get_bloom_config();
        if (bloom_config.expected_elements == 0) {
            errors.push_back("Bloom filter expected_elements cannot be zero");
        }
        if (bloom_config.false_positive_rate <= 0.0 || bloom_config.false_positive_rate >= 1.0) {
            errors.push_back("Bloom filter false_positive_rate must be between 0 and 1");
        }
        
        // اعتبارسنجی پارامترهای GPU
        auto mining_config = get_mining_config();
        if (mining_config.keys_per_batch == 0) {
            errors.push_back("Mining keys_per_batch cannot be zero");
        }
        
        // اعتبارسنجی پارامترهای عمومی
        if (get_int("system.max_memory_usage_mb", 8192) < 100) {
            errors.push_back("System max_memory_usage_mb must be at least 100 MB");
        }
        
        if (errors.empty()) {
            return true;
        } else {
            for (const auto& error : errors) {
                Logger::error("Config validation error: {}", error);
            }
            return false;
        }
    }
    
    std::vector<std::string> get_validation_errors() const {
        // برای سادگی، فقط validate را فراخوانی می‌کنیم
        validate();
        return {}; // می‌تواند لیست خطاها را برگرداند
    }
    
    bool update_config(const std::string& key, const std::string& value) {
        config_data[key] = value;
        return validate();
    }
    
    void set_defaults() {
        // مقادیر پیش‌فرض بهینه‌شده
        config_data = {
            // سیستم
            {"system.log_level", "INFO"},
            {"system.log_file", "logs/miner.log"},
            {"system.output_directory", "outputs/results"},
            {"system.max_memory_usage_mb", "8192"},
            {"system.enable_monitoring", "true"},
            {"system.monitoring_port", "8080"},
            
            // GPU
            {"gpu.batch_size", "1000000"},
            {"gpu.threads_per_block", "256"},
            {"gpu.max_concurrent_batches", "4"},
            {"gpu.use_pinned_memory", "true"},
            {"gpu.enable_concurrent_kernels", "true"},
            
            // Bloom Filter - پارامترهای بهینه‌شده
            {"bloom_filter.expected_elements", "50000000"},
            {"bloom_filter.false_positive_rate", "0.000001"}, // 1e-6
            {"bloom_filter.memory_limit_mb", "200"},
            {"bloom_filter.gpu_accelerated", "true"},
            {"bloom_filter.num_hashes", "20"},
            {"bloom_filter.seed", "1234567890"},
            
            // آدرس
            {"address.formats", "p2pkh,p2sh,bech32"},
            {"address.compressed_public_keys", "true"},
            {"address.testnet_mode", "false"},
            {"address.address_version", "0"},
            {"address.bech32_hrp", "bc"},
            
            // ماینینگ
            {"mining.keys_per_batch", "1000000"},
            {"mining.max_concurrent_batches", "8"},
            {"mining.batch_size_auto_adjust", "true"},
            {"mining.verification_enabled", "true"},
            {"mining.save_all_results", "false"},
            {"mining.save_interval_seconds", "300"},
            
            // ذخیره‌سازی
            {"storage.auto_save", "true"},
            {"storage.max_memory_results", "1000"},
            {"storage.backup_interval_hours", "24"},
            {"storage.compression_enabled", "true"},
            {"storage.output_format", "json"},
            
            // مانیتورینگ
            {"monitoring.enable_live_reporter", "true"},
            {"monitoring.report_interval_seconds", "30"},
            {"monitoring.enable_web_dashboard", "true"},
            {"monitoring.web_port", "8080"},
            {"monitoring.save_statistics", "true"},
            {"monitoring.statistics_interval_minutes", "5"},
            
            // عملکرد
            {"performance.adaptive_optimization", "true"},
            {"performance.load_balancing", "true"},
            {"performance.auto_tune_interval_minutes", "60"},
            {"performance.performance_monitoring", "true"},
            {"performance.gpu_health_check", "true"},
            
            // امنیت
            {"security.secure_random_source", "true"},
            {"security.validate_private_keys", "true"},
            {"security.checksum_verification", "true"},
            {"security.enable_sandbox", "false"},
            {"security.max_key_retries", "3"}
        };
    }

private:
    std::unordered_map<std::string, std::string> config_data;
};

// پیاده‌سازی متدهای اصلی ConfigManager
ConfigManager::ConfigManager() : impl(new Impl()) {}
ConfigManager::~ConfigManager() = default;

bool ConfigManager::load_from_file(const std::string& filename) {
    return impl->load_from_file(filename);
}

bool ConfigManager::save_to_file(const std::string& filename) const {
    return impl->save_to_file(filename);
}

bool ConfigManager::load_from_string(const std::string& config_data) {
    return impl->load_from_string(config_data);
}

ConfigManager::GPUConfig ConfigManager::get_gpu_config(int device_id) const {
    return impl->get_gpu_config(device_id);
}

ConfigManager::BloomFilterConfig ConfigManager::get_bloom_config() const {
    return impl->get_bloom_config();
}

ConfigManager::MiningConfig ConfigManager::get_mining_config() const {
    return impl->get_mining_config();
}

std::string ConfigManager::get_string(const std::string& key, const std::string& default_val) const {
    return impl->get_string(key, default_val);
}

int ConfigManager::get_int(const std::string& key, int default_val) const {
    return impl->get_int(key, default_val);
}

double ConfigManager::get_double(const std::string& key, double default_val) const {
    return impl->get_double(key, default_val);
}

bool ConfigManager::get_bool(const std::string& key, bool default_val) const {
    return impl->get_bool(key, default_val);
}

bool ConfigManager::validate() const {
    return impl->validate();
}

std::vector<std::string> ConfigManager::get_validation_errors() const {
    return impl->get_validation_errors();
}

bool ConfigManager::update_config(const std::string& key, const std::string& value) {
    return impl->update_config(key, value);
}

void ConfigManager::set_defaults() {
    impl->set_defaults();
}

} // namespace bitcoin_miner