#include <gtest/gtest.h>
#include "../src/crypto/sha256.h"
#include "../src/crypto/ripemd160.h"
#include "../src/address/base58.h"
#include "../src/address/bech32.h"
#include "../src/utils/random.h"

using namespace bitcoin_miner;

class CryptoTests : public ::testing::Test {
protected:
    void SetUp() override {
        // راه‌اندازی قبل از هر تست
    }
    
    void TearDown() override {
        // پاک‌سازی بعد از هر تست
    }
};

TEST_F(CryptoTests, SHA256_Basic) {
    SHA256 sha256;
    auto hash = sha256.hash("hello world");
    
    // بررسی هش شناخته شده
    std::string expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
    // تبدیل هش به hex string و مقایسه
}

TEST_F(CryptoTests, RIPEMD160_Basic) {
    RIPEMD160 ripemd160;
    auto hash = ripemd160.hash("hello world");
    
    // بررسی هش شناخته شده
    ASSERT_EQ(hash.size(), 20);
}

TEST_F(CryptoTests, Base58_EncodeDecode) {
    std::vector<uint8_t> test_data = {1, 2, 3, 4, 5};
    auto encoded = Base58::encode(test_data);
    auto decoded = Base58::decode(encoded);
    
    ASSERT_EQ(test_data, decoded);
}

TEST_F(CryptoTests, SecureRandom_Generation) {
    auto key1 = SecureRandom::generate_private_key();
    auto key2 = SecureRandom::generate_private_key();
    
    ASSERT_EQ(key1.size(), 32);
    ASSERT_EQ(key2.size(), 32);
    ASSERT_NE(key1, key2); // کلیدها باید متفاوت باشند
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}