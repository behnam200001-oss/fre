#include "address_factory.h"
#include "base58.h"
#include "bech32.h"
#include "../crypto/sha256.h"
#include "../crypto/ripemd160.h"
#include "../utils/format_utils.h"
#include "../utils/logger.h"

#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <sstream>
#include <iomanip>

namespace bitcoin_miner {

class AddressFactory::Impl {
public:
    Impl() : testnet(false) {}
    
    // تولید کلید خصوصی امن
    std::vector<uint8_t> generate_private_key() {
        return SecureRandom::generate_private_key();
    }
    
    // تولید کلید عمومی از کلید خصوصی
    std::vector<uint8_t> private_to_public(const std::vector<uint8_t>& private_key, bool compressed) {
        // در اینجا از پیاده‌سازی secp256k1 استفاده می‌کنیم
        std::vector<uint8_t> public_key;
        
        if (compressed) {
            public_key.resize(33);
            // پیاده‌سازی واقعی در کرنل GPU انجام می‌شود
            // این یک پیاده‌سازی ساده برای تست است
            public_key[0] = 0x02; // compressed even
            // 32 بایت باقیمانده از محاسبات بیضوی
        } else {
            public_key.resize(65);
            public_key[0] = 0x04; // uncompressed
            // 64 بایت باقیمانده
        }
        
        return public_key;
    }
    
    // تولید آدرس P2PKH
    std::string create_p2pkh_address(const std::vector<uint8_t>& public_key, bool compressed) {
        // 1. هش SHA256 + RIPEMD160 کلید عمومی
        std::vector<uint8_t> hash160 = hash160_data(public_key);
        
        // 2. افزودن version byte
        std::vector<uint8_t> payload;
        uint8_t version = testnet ? 0x6F : 0x00;
        payload.push_back(version);
        payload.insert(payload.end(), hash160.begin(), hash160.end());
        
        // 3. محاسبه checksum
        std::vector<uint8_t> checksum = calculate_checksum(payload);
        
        // 4. ساخت آدرس Base58
        payload.insert(payload.end(), checksum.begin(), checksum.end());
        return Base58::encode(payload);
    }
    
    // تولید آدرس P2SH
    std::string create_p2sh_address(const std::vector<uint8_t>& public_key, bool compressed) {
        // 1. هش کلید عمومی
        std::vector<uint8_t> hash160 = hash160_data(public_key);
        
        // 2. ساخت script (P2SH)
        std::vector<uint8_t> script;
        script.push_back(0x00); // OP_0
        script.push_back(0x14); // Push 20 bytes
        script.insert(script.end(), hash160.begin(), hash160.end());
        
        // 3. هش script
        std::vector<uint8_t> script_hash = hash160_data(script);
        
        // 4. ساخت آدرس
        std::vector<uint8_t> payload;
        uint8_t version = testnet ? 0xC4 : 0x05;
        payload.push_back(version);
        payload.insert(payload.end(), script_hash.begin(), script_hash.end());
        
        std::vector<uint8_t> checksum = calculate_checksum(payload);
        payload.insert(payload.end(), checksum.begin(), checksum.end());
        
        return Base58::encode(payload);
    }
    
    // تولید آدرس Bech32
    std::string create_bech32_address(const std::vector<uint8_t>& public_key, bool compressed) {
        // 1. هش کلید عمومی
        std::vector<uint8_t> hash160 = hash160_data(public_key);
        
        // 2. تبدیل به words (5-bit)
        std::vector<uint8_t> witness_program;
        witness_program.push_back(0x00); // witness version (0 for P2WPKH)
        witness_program.push_back(0x14); // program length (20 bytes)
        witness_program.insert(witness_program.end(), hash160.begin(), hash160.end());
        
        // 3. کدگذاری Bech32
        std::string hrp = testnet ? "tb" : "bc";
        return Bech32::encode(hrp, witness_program);
    }
    
    // اعتبارسنجی آدرس
    bool validate_address(const std::string& address) {
        if (address.empty()) return false;
        
        try {
            if (address[0] == '1' || address[0] == '3') {
                return validate_base58_address(address);
            } else if (address.substr(0, 3) == "bc1" || address.substr(0, 3) == "tb1") {
                return validate_bech32_address(address);
            }
        } catch (const std::exception& e) {
            Logger::error("Address validation error: {}", e.what());
            return false;
        }
        
        return false;
    }
    
    // تشخیص نوع آدرس
    std::string detect_address_type(const std::string& address) {
        if (address.empty()) return "unknown";
        
        if (address[0] == '1') return "p2pkh";
        if (address[0] == '3') return "p2sh";
        if (address.substr(0, 3) == "bc1") return "bech32";
        if (address.substr(0, 3) == "tb1") return "bech32_testnet";
        
        return "unknown";
    }
    
    // تبدیل کلید خصوصی به WIF
    std::string private_key_to_wif(const std::vector<uint8_t>& private_key, bool compressed, bool testnet) {
        std::vector<uint8_t> payload;
        
        // version byte
        uint8_t version = testnet ? 0xEF : 0x80;
        payload.push_back(version);
        
        // private key
        payload.insert(payload.end(), private_key.begin(), private_key.end());
        
        // compressed flag
        if (compressed) {
            payload.push_back(0x01);
        }
        
        // checksum
        std::vector<uint8_t> checksum = calculate_checksum(payload);
        payload.insert(payload.end(), checksum.begin(), checksum.end());
        
        return Base58::encode(payload);
    }
    
    // تبدیل WIF به کلید خصوصی
    std::vector<uint8_t> wif_to_private_key(const std::string& wif) {
        std::vector<uint8_t> decoded = Base58::decode_check(wif);
        
        // بررسی version byte
        uint8_t version = decoded[0];
        if (version != 0x80 && version != 0xEF) {
            throw std::runtime_error("Invalid WIF version byte");
        }
        
        // استخراج private key
        size_t key_size = (decoded.size() == 34) ? 32 : 33; // با یا بدون compressed flag
        std::vector<uint8_t> private_key(decoded.begin() + 1, decoded.begin() + 1 + key_size);
        
        return private_key;
    }
    
    void set_testnet(bool enabled) { testnet = enabled; }
    
private:
    bool testnet;
    
    std::vector<uint8_t> hash160_data(const std::vector<uint8_t>& data) {
        // SHA256 سپس RIPEMD160
        SHA256 sha256;
        auto sha256_hash = sha256.hash(data);
        
        RIPEMD160 ripemd160;
        return ripemd160.hash(sha256_hash);
    }
    
    std::vector<uint8_t> calculate_checksum(const std::vector<uint8_t>& data) {
        // double SHA256
        SHA256 sha256;
        auto first_hash = sha256.hash(data);
        auto second_hash = sha256.hash(first_hash);
        
        // 4 بایت اول
        return std::vector<uint8_t>(second_hash.begin(), second_hash.begin() + 4);
    }
    
    bool validate_base58_address(const std::string& address) {
        try {
            std::vector<uint8_t> decoded = Base58::decode_check(address);
            return !decoded.empty();
        } catch (...) {
            return false;
        }
    }
    
    bool validate_bech32_address(const std::string& address) {
        try {
            std::string hrp;
            std::vector<uint8_t> data;
            return Bech32::decode(address, hrp, data);
        } catch (...) {
            return false;
        }
    }
};

// پیاده‌سازی متدهای اصلی AddressFactory
AddressFactory::AddressFactory() : impl(new Impl()) {}
AddressFactory::~AddressFactory() = default;

AddressFactory::KeyPair AddressFactory::generate_keypair() {
    KeyPair keypair;
    keypair.private_key = impl->generate_private_key();
    keypair.public_key_compressed = impl->private_to_public(keypair.private_key, true);
    keypair.public_key_uncompressed = impl->private_to_public(keypair.private_key, false);
    return keypair;
}

std::string AddressFactory::generate_p2pkh_address(const KeyPair& keys, bool compressed, bool testnet) {
    impl->set_testnet(testnet);
    return impl->create_p2pkh_address(
        compressed ? keys.public_key_compressed : keys.public_key_uncompressed, 
        compressed
    );
}

std::string AddressFactory::generate_p2sh_address(const KeyPair& keys, bool testnet) {
    impl->set_testnet(testnet);
    return impl->create_p2sh_address(keys.public_key_compressed, true);
}

std::string AddressFactory::generate_bech32_address(const KeyPair& keys, bool testnet) {
    impl->set_testnet(testnet);
    return impl->create_bech32_address(keys.public_key_compressed, true);
}

bool AddressFactory::validate_address(const std::string& address) {
    return impl->validate_address(address);
}

std::string AddressFactory::detect_address_type(const std::string& address) {
    return impl->detect_address_type(address);
}

std::string AddressFactory::private_key_to_wif(const std::vector<uint8_t>& private_key, bool compressed, bool testnet) {
    return impl->private_key_to_wif(private_key, compressed, testnet);
}

std::vector<uint8_t> AddressFactory::wif_to_private_key(const std::string& wif) {
    return impl->wif_to_private_key(wif);
}

} // namespace bitcoin_miner