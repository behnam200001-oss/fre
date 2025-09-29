#pragma once
#ifndef ADVANCED_SECP256K1_H
#define ADVANCED_SECP256K1_H

#include <vector>
#include <cstdint>

namespace bitcoin_miner {

// تعریف ساختار ECPoint
struct ECPoint {
    uint32_t x[8];
    uint32_t y[8];
};

class AdvancedSecp256k1 {
private:
    static constexpr size_t MAX_BATCH_SIZE = 1000000;
    uint8_t* d_private_keys;
    uint8_t* d_public_keys;
    uint8_t* d_results;

public:
    AdvancedSecp256k1();
    ~AdvancedSecp256k1();
    
    std::vector<std::vector<uint8_t>> generate_public_keys(
        const std::vector<std::vector<uint8_t>>& private_keys, bool compressed = true);
    
    bool validate_private_key(const std::vector<uint8_t>& private_key);
    bool validate_keypair(const std::vector<uint8_t>& private_key, const std::vector<uint8_t>& public_key);

private:
    void cleanup();
};

} // namespace bitcoin_miner

#endif