#pragma once
#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace bitcoin_miner {

class ConfigManager {
public:
    struct GPUConfig {
        int device_id;
        bool enabled;
        size_t workload_percentage;
        size_t batch_size;
        size_t threads_per_block;
        size_t max_utilization;
    };
    
    struct BloomFilterConfig {
        size_t expected_elements;
        double false_positive_rate;
        size_t memory_limit_mb;
        bool gpu_accelerated;
    };
    
    struct MiningConfig {
        size_t keys_per_batch;
        size_t max_concurrent_batches;
        bool compressed_addresses;
        bool enable_all_formats;
        uint8_t address_version;
        bool testnet_mode;
    };
    
    ConfigManager();
    ~ConfigManager();
    
    // بارگیری و ذخیره پیکربندی
    bool load_from_file(const std::string& filename);
    bool save_to_file(const std::string& filename) const;
    bool load_from_string(const std::string& config_data);
    
    // دسترسی به تنظیمات
    GPUConfig get_gpu_config(int device_id) const;
    BloomFilterConfig get_bloom_config() const;
    MiningConfig get_mining_config() const;
    
    // تنظیمات پیشرفته
    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    int get_int(const std::string& key, int default_val = 0) const;
    double get_double(const std::string& key, double default_val = 0.0) const;
    bool get_bool(const std::string& key, bool default_val = false) const;
    
    // اعتبارسنجی پیکربندی
    bool validate() const;
    std::vector<std::string> get_validation_errors() const;
    
    // به‌روزرسانی پویا
    bool update_config(const std::string& key, const std::string& value);
    void set_defaults();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bitcoin_miner

#endif // CONFIG_MANAGER_H