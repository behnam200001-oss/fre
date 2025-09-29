#include <gtest/gtest.h>
#include <chrono>
#include "../../src/crypto/advanced_secp256k1.h"
#include "../../src/bloom/super_bloom_filter.h"
#include "../../src/utils/logger.h"

using namespace bitcoin_miner;

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize("logs/performance_test.log", LogLevel::INFO);
    }
    
    void TearDown() override {
        Logger::shutdown();
    }
};

TEST_F(PerformanceTest, Secp256k1_Batch_Performance) {
    AdvancedSecp256k1 crypto;
    
    // تولید کلیدهای خصوصی تست
    const size_t batch_size = 100000;
    std::vector<std::vector<uint8_t>> private_keys;
    for (size_t i = 0; i < batch_size; i++) {
        private_keys.push_back(SecureRandom::generate_private_key());
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // تولید کلیدهای عمومی
    auto public_keys = crypto.generate_public_keys(private_keys, true);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double keys_per_second = (batch_size * 1000.0) / duration.count();
    
    Logger::info("Secp256k1 Performance: {} keys in {} ms ({:.2f} K keys/sec)", 
                batch_size, duration.count(), keys_per_second / 1000);
    
    EXPECT_EQ(public_keys.size(), batch_size);
    EXPECT_GT(keys_per_second, 1000); // حداقل 1000 کلید در ثانیه
}

TEST_F(PerformanceTest, BloomFilter_Performance) {
    const size_t filter_size = 1000000;
    const size_t test_items = 100000;
    
    SuperBloomFilter bloom_filter(filter_size, 0.001);
    
    // افزودن آیتم‌ها به بلوم فیلتر
    auto start_add = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < filter_size; i++) {
        std::string item = "address_" + std::to_string(i);
        bloom_filter.add_address(item);
    }
    
    auto end_add = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_add - start_add);
    
    // تست بررسی آیتم‌ها
    auto start_check = std::chrono::high_resolution_clock::now();
    
    size_t found_count = 0;
    for (size_t i = 0; i < test_items; i++) {
        std::string item = "address_" + std::to_string(i % filter_size);
        if (bloom_filter.contains(item)) {
            found_count++;
        }
    }
    
    auto end_check = std::chrono::high_resolution_clock::now();
    auto check_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_check - start_check);
    
    Logger::info("Bloom Filter Performance:");
    Logger::info("  Add: {} items in {} ms ({:.2f} K items/sec)", 
                filter_size, add_duration.count(), (filter_size * 1000.0) / add_duration.count() / 1000);
    Logger::info("  Check: {} items in {} ms ({:.2f} K checks/sec)", 
                test_items, check_duration.count(), (test_items * 1000.0) / check_duration.count() / 1000);
    
    EXPECT_GT(found_count, 0);
}

TEST_F(PerformanceTest, AddressGeneration_Performance) {
    AddressFactory factory;
    const size_t batch_size = 50000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < batch_size; i++) {
        auto keypair = factory.generate_keypair();
        auto address = factory.generate_p2pkh_address(keypair, true, false);
        
        // اعتبارسنجی آدرس تولید شده
        EXPECT_TRUE(factory.validate_address(address));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double addresses_per_second = (batch_size * 1000.0) / duration.count();
    
    Logger::info("Address Generation Performance: {} addresses in {} ms ({:.2f} addresses/sec)", 
                batch_size, duration.count(), addresses_per_second);
    
    EXPECT_GT(addresses_per_second, 100); // حداقل 100 آدرس در ثانیه
}