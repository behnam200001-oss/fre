#pragma once
#ifndef SUPER_BLOOM_FILTER_H
#define SUPER_BLOOM_FILTER_H

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include "../utils/cuda_utils.h"

namespace bitcoin_miner {

struct GPU_BloomFilter {
    uint64_t* data;
    uint64_t bit_size;
    uint32_t num_hashes;
    uint64_t seed;
    uint32_t array_size;
};

class SuperBloomFilter {
public:
    // سازنده با پارامترهای بهینه‌شده
    SuperBloomFilter(size_t expected_elements = 50000000, 
                    double false_positive_rate = 0.000001, // 1e-6
                    uint64_t seed = 0x1234567890ABCDEFULL);
    ~SuperBloomFilter();

    // غیرقابل کپی
    SuperBloomFilter(const SuperBloomFilter&) = delete;
    SuperBloomFilter& operator=(const SuperBloomFilter&) = delete;

    // مدیریت چرخه حیات
    bool initialize();
    void cleanup();
    
    // افزودن آدرس به فیلتر
    void add_address(const std::string& address);
    void add_address_batch(const std::vector<std::string>& addresses);
    
    // بررسی وجود آدرس
    bool contains(const std::string& address) const;
    std::vector<bool> contains_batch(const std::vector<std::string>& addresses) const;
    
    // نسخه GPU-accelerated
    GPU_BloomFilter get_gpu_representation() const;
    void update_gpu_representation();
    
    // آمار و اطلاعات
    size_t get_bit_size() const { return bit_size; }
    size_t get_byte_size() const { return (bit_size + 7) / 8; }
    size_t get_num_hashes() const { return num_hashes; }
    size_t get_count() const { return element_count; }
    double get_false_positive_rate() const;
    
    // محاسبات تئوری
    static size_t calculate_optimal_bit_size(size_t n, double p);
    static uint32_t calculate_optimal_num_hashes(size_t n, size_t m);
    
    // مدیریت حافظه
    bool load_from_file(const std::string& filename);
    bool save_to_file(const std::string& filename) const;
    
    // اعتبارسنجی
    bool validate() const;

private:
    // پارامترهای بلوم فیلتر
    size_t expected_elements;
    double target_false_positive_rate;
    uint64_t seed;
    
    // وضعیت فعلی
    size_t bit_size;
    uint32_t num_hashes;
    size_t array_size; // تعداد uint64_t
    std::vector<uint64_t> bit_array;
    std::atomic<size_t> element_count{0};
    
    // نسخه GPU
    uint64_t* d_bit_array{nullptr};
    bool gpu_initialized{false};
    
    // ثابت‌ها برای double hashing
    static constexpr uint64_t SEED1 = 0x1234567890ABCDEFULL;
    static constexpr uint64_t SEED2 = 0xFEDCBA0987654321ULL;
    
    // متدهای داخلی
    void calculate_optimal_parameters();
    std::vector<uint64_t> calculate_hashes(const std::string& data) const;
    uint64_t murmurhash3_64(const std::string& data, uint64_t seed) const;
    uint64_t double_hash(const std::string& data, uint32_t hash_index) const;
    
    // متدهای GPU
    bool initialize_gpu();
    void cleanup_gpu();
};

} // namespace bitcoin_miner

#endif // SUPER_BLOOM_FILTER_H