#pragma once
#ifndef RESULT_MANAGER_H
#define RESULT_MANAGER_H

#include "../core/mining_result.h"
#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <atomic>
#include <memory>

namespace bitcoin_miner {

class ResultManager {
public:
    ResultManager();
    ~ResultManager();
    
    // پیکربندی
    bool initialize(const std::string& output_dir = "outputs/results");
    void set_auto_save(bool enabled);
    void set_max_memory_usage(size_t max_mb);
    
    // مدیریت نتایج
    void save_result(const MiningResult& result);
    void save_results_batch(const std::vector<MiningResult>& results);
    
    // بازیابی و جستجو
    std::vector<MiningResult> load_results(const std::string& filter = "") const;
    bool contains_result(const std::string& address) const;
    
    // آمار و اطلاعات
    size_t get_result_count() const;
    size_t get_database_size() const;
    void print_statistics() const;
    
    // مدیریت فایل‌ها
    bool export_to_csv(const std::string& filename) const;
    bool export_to_json(const std::string& filename) const;
    bool backup_database(const std::string& backup_path) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // غیرقابل کپی
    ResultManager(const ResultManager&) = delete;
    ResultManager& operator=(const ResultManager&) = delete;
};

} // namespace bitcoin_miner

#endif // RESULT_MANAGER_H