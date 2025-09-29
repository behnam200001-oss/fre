#include "super_bloom_filter.h"
#include "../utils/logger.h"
#include "../utils/cuda_utils.h"
#include "../crypto/sha256.h"
#include "../crypto/ripemd160.h"
#include "../address/base58.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>

namespace bitcoin_miner {

// پیاده‌سازی سازنده با پارامترهای بهینه‌شده
SuperBloomFilter::SuperBloomFilter(size_t expected_elements, 
                                 double false_positive_rate, 
                                 uint64_t seed)
    : expected_elements(expected_elements)
    , target_false_positive_rate(false_positive_rate)
    , seed(seed) {
    
    calculate_optimal_parameters();
    Logger::info("Bloom Filter initialized: bits={}, hashes={}, target_fp={}",
                bit_size, num_hashes, target_false_positive_rate);
}

SuperBloomFilter::~SuperBloomFilter() {
    cleanup();
}

void SuperBloomFilter::calculate_optimal_parameters() {
    // محاسبه پارامترهای بهینه بر اساس فرمول‌های تئوری
    bit_size = calculate_optimal_bit_size(expected_elements, target_false_positive_rate);
    num_hashes = calculate_optimal_num_hashes(expected_elements, bit_size);
    array_size = (bit_size + 63) / 64; // تعداد uint64_t مورد نیاز
    
    Logger::debug("Bloom parameters: n={}, p={}, m={}, k={}", 
                 expected_elements, target_false_positive_rate, bit_size, num_hashes);
}

size_t SuperBloomFilter::calculate_optimal_bit_size(size_t n, double p) {
    if (n == 0 || p <= 0.0 || p >= 1.0) {
        return 8 * 1024 * 1024; // 8MB fallback
    }
    
    double ln2 = log(2);
    double numerator = -n * log(p);
    double denominator = ln2 * ln2;
    
    size_t m = static_cast<size_t>(numerator / denominator);
    
    // گرد کردن به نزدیکترین مضرب 64
    m = ((m + 63) / 64) * 64;
    
    return m;
}

uint32_t SuperBloomFilter::calculate_optimal_num_hashes(size_t n, size_t m) {
    if (n == 0) return 7;
    
    uint32_t k = static_cast<uint32_t>((static_cast<double>(m) / n) * log(2));
    k = std::max(1u, k);
    k = std::min(30u, k); // محدود کردن به 30 برای عملکرد
    
    return k;
}

bool SuperBloomFilter::initialize() {
    try {
        // تخصیص حافظه میزبان
        bit_array.resize(array_size, 0);
        
        // راه‌اندازی GPU
        if (!initialize_gpu()) {
            Logger::error("Failed to initialize GPU Bloom filter");
            return false;
        }
        
        Logger::info("Bloom filter initialized successfully: {} bits ({} MB)", 
                    bit_size, get_byte_size() / (1024 * 1024));
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Bloom filter initialization failed: {}", e.what());
        return false;
    }
}

void SuperBloomFilter::cleanup() {
    cleanup_gpu();
    bit_array.clear();
    element_count = 0;
}

bool SuperBloomFilter::initialize_gpu() {
    try {
        if (array_size == 0) {
            Logger::error("Cannot initialize GPU: array size is zero");
            return false;
        }
        
        // تخصیص حافظه GPU
        size_t bytes_needed = array_size * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&d_bit_array, bytes_needed));
        
        // کپی داده‌های اولیه
        CUDA_CHECK(cudaMemcpy(d_bit_array, bit_array.data(), 
                             bytes_needed, cudaMemcpyHostToDevice));
        
        gpu_initialized = true;
        Logger::debug("GPU Bloom filter initialized: {} bytes", bytes_needed);
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("GPU Bloom initialization failed: {}", e.what());
        if (d_bit_array) {
            cudaFree(d_bit_array);
            d_bit_array = nullptr;
        }
        return false;
    }
}

void SuperBloomFilter::cleanup_gpu() {
    if (d_bit_array) {
        CUDA_CHECK(cudaFree(d_bit_array));
        d_bit_array = nullptr;
    }
    gpu_initialized = false;
}

// پیاده‌سازی کامل MurmurHash3
uint64_t SuperBloomFilter::murmurhash3_64(const void* key, int len, uint64_t seed) const {
    const uint64_t m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint8_t* data = (const uint8_t*)key;
    const uint8_t* end = data + (len - (len & 7));

    while (data != end) {
        uint64_t k;
        memcpy(&k, data, sizeof(k));
        data += sizeof(k);

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    // پردازش بایت‌های باقیمانده
    switch (len & 7) {
        case 7: h ^= uint64_t(data[6]) << 48;
        case 6: h ^= uint64_t(data[5]) << 40;
        case 5: h ^= uint64_t(data[4]) << 32;
        case 4: h ^= uint64_t(data[3]) << 24;
        case 3: h ^= uint64_t(data[2]) << 16;
        case 2: h ^= uint64_t(data[1]) << 8;
        case 1: h ^= uint64_t(data[0]);
                h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

// پیاده‌سازی double hashing بهینه‌شده
uint64_t SuperBloomFilter::double_hash(const std::string& data, uint32_t hash_index) const {
    uint64_t h1 = murmurhash3_64(data, seed);
    uint64_t h2 = murmurhash3_64(data, seed ^ 0xFEDCBA0987654321ULL); // seed متفاوت
    
    // ترکیب هوشمند هش‌ها برای توزیع بهتر
    return h1 + hash_index * h2 + (hash_index * hash_index); // اضافه کردن ترم درجه دوم
}

// بهبود افزودن دسته‌ای با sync خودکار GPU
void SuperBloomFilter::add_address_batch(const std::vector<std::string>& addresses) {
    std::lock_guard<std::mutex> lock(bloom_mutex);
    
    for (const auto& address : addresses) {
        if (address.empty()) continue;
        
        for (uint32_t i = 0; i < num_hashes; i++) {
            uint64_t hash = double_hash(address, i);
            uint64_t bit_index = hash % bit_size;
            uint64_t word_index = bit_index / 64;
            uint64_t bit_mask = 1ULL << (bit_index % 64);
            
            // بررسی overflow
            if (word_index >= array_size) {
                Logger::error("Bloom filter index overflow: {} >= {}", word_index, array_size);
                continue;
            }
            
            bit_array[word_index] |= bit_mask;
        }
        
        element_count++;
    }
    
    // همیشه پس از افزودن دسته‌ای، GPU را sync کنید
    if (gpu_initialized) {
        update_gpu_representation();
    }
}

// افزودن آدرس واحد
void SuperBloomFilter::add_address(const std::string& address) {
    if (address.empty()) return;
    
    std::lock_guard<std::mutex> lock(bloom_mutex);
    
    for (uint32_t i = 0; i < num_hashes; i++) {
        uint64_t hash = double_hash(address, i);
        uint64_t bit_index = hash % bit_size;
        uint64_t word_index = bit_index / 64;
        uint64_t bit_mask = 1ULL << (bit_index % 64);
        
        if (word_index >= array_size) {
            Logger::error("Bloom filter index overflow: {} >= {}", word_index, array_size);
            continue;
        }
        
        bit_array[word_index] |= bit_mask;
    }
    
    element_count++;
    
    // به‌روزرسانی نسخه GPU اگر لازم باشد
    if (gpu_initialized && element_count % 1000 == 0) {
        update_gpu_representation();
    }
}

// بررسی وجود آدرس
bool SuperBloomFilter::contains(const std::string& address) const {
    if (address.empty()) return false;
    
    std::lock_guard<std::mutex> lock(bloom_mutex);
    
    for (uint32_t i = 0; i < num_hashes; i++) {
        uint64_t hash = double_hash(address, i);
        uint64_t bit_index = hash % bit_size;
        uint64_t word_index = bit_index / 64;
        uint64_t bit_mask = 1ULL << (bit_index % 64);
        
        if (word_index >= array_size) {
            return false;
        }
        
        if ((bit_array[word_index] & bit_mask) == 0) {
            return false;
        }
    }
    
    return true;
}

// بررسی دسته‌ای
std::vector<bool> SuperBloomFilter::contains_batch(const std::vector<std::string>& addresses) const {
    std::vector<bool> results;
    results.reserve(addresses.size());
    
    std::lock_guard<std::mutex> lock(bloom_mutex);
    
    for (const auto& address : addresses) {
        bool found = true;
        
        for (uint32_t i = 0; i < num_hashes; i++) {
            uint64_t hash = double_hash(address, i);
            uint64_t bit_index = hash % bit_size;
            uint64_t word_index = bit_index / 64;
            uint64_t bit_mask = 1ULL << (bit_index % 64);
            
            if (word_index >= array_size || (bit_array[word_index] & bit_mask) == 0) {
                found = false;
                break;
            }
        }
        
        results.push_back(found);
    }
    
    return results;
}

// به‌روزرسانی نمایش GPU
void SuperBloomFilter::update_gpu_representation() {
    if (!gpu_initialized || d_bit_array == nullptr) return;
    
    try {
        size_t bytes_needed = array_size * sizeof(uint64_t);
        CUDA_CHECK(cudaMemcpy(d_bit_array, bit_array.data(), 
                             bytes_needed, cudaMemcpyHostToDevice));
    } catch (const std::exception& e) {
        Logger::error("Failed to update GPU Bloom filter: {}", e.what());
    }
}

// دریافت نمایش GPU
GPU_BloomFilter SuperBloomFilter::get_gpu_representation() const {
    GPU_BloomFilter gpu_bloom;
    gpu_bloom.data = d_bit_array;
    gpu_bloom.bit_size = bit_size;
    gpu_bloom.num_hashes = num_hashes;
    gpu_bloom.seed = seed;
    gpu_bloom.array_size = array_size;
    return gpu_bloom;
}

// محاسبه نرخ false positive تئوری
double SuperBloomFilter::get_false_positive_rate() const {
    if (element_count == 0) return 0.0;
    
    double k = static_cast<double>(num_hashes);
    double n = static_cast<double>(element_count);
    double m = static_cast<double>(bit_size);
    
    // فرمول تئوری نرخ false positive
    return pow(1.0 - exp(-k * n / m), k);
}

// محاسبه اشباع Bloom Filter
double SuperBloomFilter::get_saturation() const {
    if (bit_size == 0) return 0.0;
    
    uint64_t bits_set = 0;
    for (size_t i = 0; i < array_size; i++) {
        bits_set += __builtin_popcountll(bit_array[i]);
    }
    
    return static_cast<double>(bits_set) / bit_size;
}

// مدیریت فایل
bool SuperBloomFilter::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Failed to open Bloom filter file: {}", filename);
        return false;
    }
    
    try {
        // خواندن هدر
        size_t saved_bit_size, saved_element_count;
        uint32_t saved_num_hashes;
        uint64_t saved_seed;
        
        file.read(reinterpret_cast<char*>(&saved_bit_size), sizeof(saved_bit_size));
        file.read(reinterpret_cast<char*>(&saved_element_count), sizeof(saved_element_count));
        file.read(reinterpret_cast<char*>(&saved_num_hashes), sizeof(saved_num_hashes));
        file.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed));
        
        // بررسی سازگاری
        if (saved_bit_size != bit_size || saved_num_hashes != num_hashes) {
            Logger::error("Bloom filter parameters mismatch");
            return false;
        }
        
        // خواندن داده‌ها
        file.read(reinterpret_cast<char*>(bit_array.data()), array_size * sizeof(uint64_t));
        
        element_count = saved_element_count;
        
        // به‌روزرسانی GPU
        if (gpu_initialized) {
            update_gpu_representation();
        }
        
        Logger::info("Bloom filter loaded from file: {} elements", element_count);
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Error loading Bloom filter: {}", e.what());
        return false;
    }
}

bool SuperBloomFilter::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Failed to create Bloom filter file: {}", filename);
        return false;
    }
    
    try {
        // نوشتن هدر
        file.write(reinterpret_cast<const char*>(&bit_size), sizeof(bit_size));
        file.write(reinterpret_cast<const char*>(&element_count), sizeof(element_count));
        file.write(reinterpret_cast<const char*>(&num_hashes), sizeof(num_hashes));
        file.write(reinterpret_cast<const char*>(&seed), sizeof(seed));
        
        // نوشتن داده‌ها
        file.write(reinterpret_cast<const char*>(bit_array.data()), array_size * sizeof(uint64_t));
        
        Logger::info("Bloom filter saved to file: {} elements", element_count);
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Error saving Bloom filter: {}", e.what());
        return false;
    }
}

// اعتبارسنجی
bool SuperBloomFilter::validate() const {
    if (bit_size == 0 || array_size == 0) {
        Logger::error("Bloom filter not properly initialized");
        return false;
    }
    
    if (num_hashes == 0 || num_hashes > 50) {
        Logger::error("Invalid number of hash functions: {}", num_hashes);
        return false;
    }
    
    // بررسی اشباع
    double saturation = get_saturation();
    if (saturation > 0.95) {
        Logger::warning("Bloom filter highly saturated: {:.1f}%", saturation * 100);
    }
    
    return true;
}

} // namespace bitcoin_miner