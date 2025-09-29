#pragma once
#ifndef MINING_RESULT_H
#define MINING_RESULT_H

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

namespace bitcoin_miner {

struct MiningResult {
    // کلید خصوصی
    std::string private_key_hex;
    std::string private_key_wif;
    std::string private_key_wif_compressed;
    
    // کلید عمومی
    std::string public_key_compressed_hex;
    std::string public_key_uncompressed_hex;
    
    // آدرس‌ها در فرمت‌های مختلف
    std::string address_p2pkh;
    std::string address_p2sh;
    std::string address_bech32;
    std::string address_bech32m;
    
    // متادیتا
    std::string address_type;
    uint64_t timestamp;
    uint64_t nonce;
    bool is_valid;
    bool verified;
    
    // اطلاعات اضافی
    std::string source_batch;
    uint32_t gpu_device_id;
    std::string worker_thread_id;
    
    // سازنده‌ها
    MiningResult();
    MiningResult(const std::string& priv_key_hex, 
                const std::string& pub_key_compressed,
                const std::string& address);
    
    // اعتبارسنجی
    bool validate() const;
    
    // فرمت‌های خروجی
    std::string to_json() const;
    std::string to_csv() const;
    std::string to_human_readable() const;
    
    // مقایسه
    bool operator==(const MiningResult& other) const;
    bool operator<(const MiningResult& other) const;
};

// مجموعه نتایج
struct MiningResultsBatch {
    std::vector<MiningResult> results;
    uint64_t batch_id;
    uint64_t start_nonce;
    uint64_t end_nonce;
    uint32_t gpu_device_id;
    std::chrono::system_clock::time_point processing_time;
    
    // آمار
    uint64_t total_processed;
    uint64_t valid_results;
    uint64_t false_positives;
    double processing_speed_kps;
    
    // متدهای مدیریتی
    MiningResultsBatch();
    void clear();
    void add_result(const MiningResult& result);
    size_t size() const;
    bool empty() const;
    double get_efficiency() const;
    
    // ذخیره و بارگذاری
    bool save_to_file(const std::string& filename) const;
    bool load_from_file(const std::string& filename);
};

// نتیجه پردازش GPU
struct GPUProcessingResult {
    uint32_t device_id;
    uint64_t keys_processed;
    uint64_t matches_found;
    uint64_t false_positives;
    double processing_time_ms;
    double utilization;
    size_t memory_used;
    
    GPUProcessingResult();
    void reset();
    double get_kps() const;
    std::string to_string() const;
};

} // namespace bitcoin_miner

#endif // MINING_RESULT_H