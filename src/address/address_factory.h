#pragma once
#ifndef ADDRESS_FACTORY_H
#define ADDRESS_FACTORY_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace bitcoin_miner {

class AddressFactory {
public:
    struct KeyPair {
        std::vector<uint8_t> private_key;
        std::vector<uint8_t> public_key_compressed;
        std::vector<uint8_t> public_key_uncompressed;
    };
    
    AddressFactory();
    ~AddressFactory();
    
    // تولید کلید
    KeyPair generate_keypair();
    
    // تولید آدرس در فرمت‌های مختلف
    std::string generate_p2pkh_address(const KeyPair& keys, bool compressed = true, bool testnet = false);
    std::string generate_p2sh_address(const KeyPair& keys, bool testnet = false);
    std::string generate_bech32_address(const KeyPair& keys, bool testnet = false);
    
    // اعتبارسنجی
    bool validate_address(const std::string& address);
    std::string detect_address_type(const std::string& address);
    
    // تبدیل فرمت‌ها
    std::string private_key_to_wif(const std::vector<uint8_t>& private_key, bool compressed = true, bool testnet = false);
    std::vector<uint8_t> wif_to_private_key(const std::string& wif);
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // غیرقابل کپی
    AddressFactory(const AddressFactory&) = delete;
    AddressFactory& operator=(const AddressFactory&) = delete;
};

} // namespace bitcoin_miner

#endif // ADDRESS_FACTORY_H