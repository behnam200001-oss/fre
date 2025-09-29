#pragma once
#ifndef ADVANCED_BECH32_H
#define ADVANCED_BECH32_H

#include <string>
#include <vector>
#include <cstdint>

namespace bitcoin_miner {

class Bech32 {
public:
    // کدگذاری داده‌ها به Bech32
    static std::string encode(const std::string& hrp, 
                             const std::vector<uint8_t>& data,
                             uint8_t witness_version = 0);
    
    // دیکد کردن Bech32
    static bool decode(const std::string& bech32, std::string& hrp, 
                      std::vector<uint8_t>& data);
    
    // تبدیل داده‌ها به words (5-bit)
    static std::vector<uint8_t> convert_bits(const std::vector<uint8_t>& data,
                                           size_t from_bits, size_t to_bits,
                                           bool pad = true);
    
    // اعتبارسنجی آدرس Bech32
    static bool validate(const std::string& bech32);
    
    // تشخیص نوع آدرس (mainnet/testnet)
    static bool is_mainnet(const std::string& bech32);
    static bool is_testnet(const std::string& bech32);
    
private:
    static const char* CHARSET;
    static const int8_t CHARSET_REV[128];
    
    static uint32_t polymod(const std::vector<uint8_t>& values);
    static std::vector<uint8_t> expand_hrp(const std::string& hrp);
    static bool verify_checksum(const std::string& hrp, 
                               const std::vector<uint8_t>& data);
    static std::vector<uint8_t> create_checksum(const std::string& hrp,
                                               const std::vector<uint8_t>& data);
};

} // namespace bitcoin_miner

#endif // ADVANCED_BECH32_H