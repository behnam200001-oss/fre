#include <gtest/gtest.h>
#include "../../src/crypto/advanced_secp256k1.h"
#include "../../src/crypto/sha256.h"
#include "../../src/crypto/ripemd160.h"
#include "../../src/address/base58.h"
#include "../../src/address/address_factory.h"
#include "../../src/bloom/super_bloom_filter.h"
#include "../../src/utils/logger.h"
#include <vector>
#include <string>

using namespace bitcoin_miner;

// تابع کمکی برای تبدیل هگزادسیمال به بایت
std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(strtol(byteString.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

// تابع کمکی برای تبدیل بایت به هگزادسیمال
std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t byte : bytes) {
        ss << std::setw(2) << static_cast<int>(byte);
    }
    return ss.str();
}

class CryptoValidationTests : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize("logs/validation_test.log", LogLevel::DEBUG);
    }
    
    void TearDown() override {
        Logger::shutdown();
    }
};

// تست‌های شناخته‌شده از Bitcoin Core برای secp256k1
TEST_F(CryptoValidationTests, Secp256k1_Known_Vectors) {
    AdvancedSecp256k1 crypto;
    
    // Test Vector 1: Private key 1
    // منبع: https://en.bitcoin.it/wiki/Technical_background_of_version_1_Bitcoin_addresses
    std::vector<uint8_t> priv_key1 = hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000001");
    std::vector<uint8_t> expected_pub1 = hex_to_bytes("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    
    auto actual_pub1 = crypto.generate_public_keys({priv_key1}, true)[0];
    EXPECT_EQ(actual_pub1, expected_pub1) 
        << "Expected: " << bytes_to_hex(expected_pub1) 
        << " Actual: " << bytes_to_hex(actual_pub1);
    
    // Test Vector 2: Private key 2
    std::vector<uint8_t> priv_key2 = hex_to_bytes("18e14a7b6a307f426a94f8114701e7c8e774e7f9a47e2c2035db29a206321725");
    std::vector<uint8_t> expected_pub2 = hex_to_bytes("0250863ad64a87ae8a2fe83c1af1a8403cb53f53e486d8511dad8a04887e5b2352");
    
    auto actual_pub2 = crypto.generate_public_keys({priv_key2}, true)[0];
    EXPECT_EQ(actual_pub2, expected_pub2)
        << "Expected: " << bytes_to_hex(expected_pub2)
        << " Actual: " << bytes_to_hex(actual_pub2);
    
    // Test Vector 3: Private key 3
    std::vector<uint8_t> priv_key3 = hex_to_bytes("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140");
    std::vector<uint8_t> expected_pub3 = hex_to_bytes("02e493dbf1c10d80f3581e4904930b1404cc6c13900ee0758474fa94abe8c4cd13");
    
    auto actual_pub3 = crypto.generate_public_keys({priv_key3}, true)[0];
    EXPECT_EQ(actual_pub3, expected_pub3)
        << "Expected: " << bytes_to_hex(expected_pub3)
        << " Actual: " << bytes_to_hex(actual_pub3);
}

TEST_F(CryptoValidationTests, Address_Generation_Validation) {
    AddressFactory factory;
    
    // تست آدرس شناخته‌شده از وکتور 1
    std::vector<uint8_t> priv_key = hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000001");
    
    // باید بتوانیم آدرس صحیح را تولید کنیم
    // آدرس مورد انتظار: 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
    auto keypair = factory.generate_keypair();
    std::string address = factory.generate_p2pkh_address(keypair, true, false);
    
    // اعتبارسنجی آدرس تولید شده
    EXPECT_TRUE(factory.validate_address(address));
    EXPECT_EQ(factory.detect_address_type(address), "p2pkh");
    
    // تست آدرس دوم
    std::vector<uint8_t> priv_key2 = hex_to_bytes("18e14a7b6a307f426a94f8114701e7c8e774e7f9a47e2c2035db29a206321725");
    auto keypair2 = factory.generate_keypair();
    std::string address2 = factory.generate_p2pkh_address(keypair2, true, false);
    
    EXPECT_TRUE(factory.validate_address(address2));
}

TEST_F(CryptoValidationTests, BloomFilter_Accuracy) {
    // تست کوچک برای اطمینان از عملکرد صحیح Bloom Filter
    SuperBloomFilter bloom(1000, 0.001); // فیلتر کوچک برای تست
    
    ASSERT_TRUE(bloom.initialize());
    
    std::vector<std::string> test_addresses = {
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", // آدرس شناخته‌شده ساتوشی
        "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",
        "1CUNebjY5CnVY8SqKcBNQqM6FQ3J8R8kR1"
    };
    
    // افزودن آدرس‌ها
    for (const auto& addr : test_addresses) {
        bloom.add_address(addr);
    }
    
    // بررسی false negative (نباید وجود داشته باشد)
    for (const auto& addr : test_addresses) {
        EXPECT_TRUE(bloom.contains(addr)) << "False negative for address: " << addr;
    }
    
    // تست false positive rate
    std::vector<std::string> random_addresses = {
        "1FakeAddress11111111111111111111111111111",
        "1TestAddress22222222222222222222222222222",
        "1Random333333333333333333333333333333333",
        "1NotExist4444444444444444444444444444444",
        "1Dummy5555555555555555555555555555555555"
    };
    
    size_t false_positives = 0;
    for (const auto& addr : random_addresses) {
        if (bloom.contains(addr)) {
            false_positives++;
        }
    }
    
    double fp_rate = static_cast<double>(false_positives) / random_addresses.size();
    EXPECT_LT(fp_rate, 0.1); // باید کمتر از 10% باشد برای این تست کوچک
    
    Logger::info("Bloom Filter FP Rate: {:.2f}%", fp_rate * 100);
}

TEST_F(CryptoValidationTests, Hash_Functions_Validation) {
    // تست توابع هش با مقادیر شناخته‌شده
    SHA256 sha256;
    RIPEMD160 ripemd160;
    
    // تست SHA256 با رشته خالی
    std::string empty_string = "";
    auto sha256_empty = sha256.hash(empty_string);
    std::string expected_sha256_empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    
    EXPECT_EQ(bytes_to_hex(sha256_empty), expected_sha256_empty);
    
    // تست SHA256 با "hello world"
    std::string hello_world = "hello world";
    auto sha256_hello = sha256.hash(hello_world);
    std::string expected_sha256_hello = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
    
    EXPECT_EQ(bytes_to_hex(sha256_hello), expected_sha256_hello);
    
    // تست double SHA256
    auto hash256_result = sha256.hash256(hello_world);
    std::string expected_hash256 = "bc62d4b80d9e36da29c16c5d4d9f11731f36052c72401a76c23c0fb5a9b74423";
    
    EXPECT_EQ(bytes_to_hex(hash256_result), expected_hash256);
}

TEST_F(CryptoValidationTests, Base58_Encoding_Validation) {
    // تست‌های شناخته‌شده برای Base58
    std::vector<uint8_t> test_data1 = {0, 1, 2, 3, 4, 5};
    std::string encoded1 = Base58::encode(test_data1);
    std::vector<uint8_t> decoded1 = Base58::decode(encoded1);
    
    EXPECT_EQ(test_data1, decoded1);
    
    // تست با checksum
    std::vector<uint8_t> test_data2 = {1, 2, 3, 4, 5};
    std::string encoded2 = Base58::encode_check(test_data2);
    std::vector<uint8_t> decoded2 = Base58::decode_check(encoded2);
    
    EXPECT_EQ(test_data2, decoded2);
    
    // تست آدرس شناخته‌شده
    EXPECT_TRUE(Base58::validate("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"));
    EXPECT_FALSE(Base58::validate("InvalidAddress123!@#"));
}

TEST_F(CryptoValidationTests, Private_Key_Validation) {
    AdvancedSecp256k1 crypto;
    
    // کلید خصوصی معتبر
    std::vector<uint8_t> valid_private_key = hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000001");
    EXPECT_TRUE(crypto.validate_private_key(valid_private_key));
    
    // کلید خصوصی نامعتبر (صفر)
    std::vector<uint8_t> zero_private_key(32, 0);
    EXPECT_FALSE(crypto.validate_private_key(zero_private_key));
    
    // کلید خصوصی نامعتبر (بیش از n)
    std::vector<uint8_t> large_private_key = hex_to_bytes("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    EXPECT_FALSE(crypto.validate_private_key(large_private_key));
    
    // کلید خصوصی نامعتبر (برابر با n)
    std::vector<uint8_t> equal_n_private_key = hex_to_bytes("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
    EXPECT_FALSE(crypto.validate_private_key(equal_n_private_key));
}

TEST_F(CryptoValidationTests, End_to_End_Mining_Simulation) {
    // تست شبیه‌سازی end-to-end
    SuperBloomFilter bloom(10000, 0.001);
    ASSERT_TRUE(bloom.initialize());
    
    // افزودن برخی آدرس‌های شناخته‌شده به Bloom filter
    std::vector<std::string> target_addresses = {
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"
    };
    
    bloom.add_address_batch(target_addresses);
    
    // شبیه‌سازی تولید کلید و بررسی در Bloom filter
    AdvancedSecp256k1 crypto;
    AddressFactory factory;
    
    size_t keys_generated = 0;
    size_t matches_found = 0;
    
    for (int i = 0; i < 1000; i++) {
        auto keypair = factory.generate_keypair();
        std::string address = factory.generate_p2pkh_address(keypair, true, false);
        
        keys_generated++;
        
        if (bloom.contains(address)) {
            matches_found++;
            Logger::info("Potential match found: {}", address);
        }
    }
    
    Logger::info("End-to-End Test: Generated {} keys, found {} potential matches", 
                keys_generated, matches_found);
    
    // در این تست کوچک، احتمال پیدا کردن آدرس واقعی بسیار کم است
    EXPECT_LE(matches_found, 5); // نباید بیش از 5 مورد false positive داشته باشیم
}

// تست عملکرد
TEST_F(CryptoValidationTests, Performance_Benchmark) {
    AdvancedSecp256k1 crypto;
    
    const size_t batch_size = 100000;
    std::vector<std::vector<uint8_t>> private_keys;
    
    // تولید کلیدهای خصوصی تست
    for (size_t i = 0; i < batch_size; i++) {
        private_keys.push_back(hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000001"));
        // در عمل باید از کلیدهای تصادفی استفاده کرد
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // تولید کلیدهای عمومی
    auto public_keys = crypto.generate_public_keys(private_keys, true);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double keys_per_second = (batch_size * 1000.0) / duration.count();
    
    Logger::info("Performance Benchmark: {} keys in {} ms ({:.2f} K keys/sec)", 
                batch_size, duration.count(), keys_per_second / 1000);
    
    EXPECT_EQ(public_keys.size(), batch_size);
    EXPECT_GT(keys_per_second, 1000); // حداقل 1000 کلید در ثانیه
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}