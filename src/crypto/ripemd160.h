#pragma once
#ifndef ADVANCED_RIPEMD160_H
#define ADVANCED_RIPEMD160_H

#include <cstdint>
#include <array>
#include <vector>
#include <string>

namespace bitcoin_miner {

class RIPEMD160 {
public:
    static constexpr size_t HASH_SIZE = 20;
    
    RIPEMD160();
    
    // هش کردن داده‌ها
    std::array<uint8_t, HASH_SIZE> hash(const uint8_t* data, size_t length);
    std::array<uint8_t, HASH_SIZE> hash(const std::vector<uint8_t>& data);
    std::array<uint8_t, HASH_SIZE> hash(const std::string& data);
    
    // متدهای incremental
    void init();
    void update(const uint8_t* data, size_t length);
    void finalize(std::array<uint8_t, HASH_SIZE>& output);
    
private:
    uint32_t state[5];
    uint64_t bit_count;
    uint8_t buffer[64];
    size_t buffer_length;
    
    void transform(const uint8_t* block);
    void pad();
    
    // توابع کمکی
    uint32_t f(uint32_t x, uint32_t y, uint32_t z, int round) {
        switch (round) {
            case 0: return x ^ y ^ z;
            case 1: return (x & y) | (~x & z);
            case 2: return (x | ~y) ^ z;
            case 3: return (x & z) | (y & ~z);
            case 4: return x ^ (y | ~z);
            default: return 0;
        }
    }
    
    // ثابت‌ها
    static constexpr uint32_t K[5] = {
        0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E
    };
    
    static constexpr uint32_t K2[5] = {
        0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000
    };
};

} // namespace bitcoin_miner

#endif // ADVANCED_RIPEMD160_H