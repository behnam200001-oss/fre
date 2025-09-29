#pragma once
#ifndef ADVANCED_RANDOM_H
#define ADVANCED_RANDOM_H

#include <cstdint>
#include <vector>
#include <random>
#include <atomic>
#include <mutex>

namespace bitcoin_miner {

class SecureRandom {
public:
    SecureRandom();
    ~SecureRandom();
    
    // تولید اعداد تصادفی امن
    static uint32_t next_uint32();
    static uint64_t next_uint64();
    static void next_bytes(uint8_t* buffer, size_t length);
    static std::vector<uint8_t> next_bytes(size_t length);
    
    // تولید کلید خصوصی امن
    static std::vector<uint8_t> generate_private_key();
    
    // seed کردن سیستم تصادفی
    static void seed(uint64_t seed_value);
    static void seed_from_system();
    
    // بررسی کیفیت entropy
    static bool has_sufficient_entropy();
    
private:
    static std::mutex mutex_;
    static std::random_device rd_;
    static std::mt19937_64 generator_;
    static std::atomic<bool> seeded_;
    
    static void ensure_seeded();
};

// نسخه GPU-optimized برای تولید اعداد تصادفی
class GPURandom {
public:
    static void initialize();
    static void cleanup();
    
    // تولید کلیدهای خصوصی دسته‌ای روی GPU
    static void generate_private_keys_batch(uint8_t* keys_buffer, size_t batch_size);
    
    // راه‌اندازی حالت‌های تصادفی برای کرنل
    static void setup_random_states(void** states_ptr, size_t num_states, uint64_t seed);
    
private:
    static bool initialized_;
};

} // namespace bitcoin_miner

#endif // ADVANCED_RANDOM_H