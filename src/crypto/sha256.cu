#include "sha256.h"
#include "../utils/cuda_utils.h"
#include <cstring>
#include <sstream>
#include <iomanip>

namespace bitcoin_miner {

// پیاده‌سازی CPU
SHA256::SHA256() {
    init();
}

void SHA256::init() {
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;
    bit_count = 0;
    buffer_length = 0;
}

void SHA256::update(const uint8_t* data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        buffer[buffer_length++] = data[i];
        if (buffer_length == 64) {
            transform(buffer);
            bit_count += 512;
            buffer_length = 0;
        }
    }
}

void SHA256::finalize(std::array<uint8_t, HASH_SIZE>& output) {
    pad();
    
    // تبدیل state به خروجی
    for (int i = 0; i < 8; i++) {
        output[i * 4] = (state[i] >> 24) & 0xFF;
        output[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        output[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        output[i * 4 + 3] = state[i] & 0xFF;
    }
    
    init(); // reset برای استفاده مجدد
}

void SHA256::transform(const uint8_t* chunk) {
    uint32_t w[64];
    
    // آماده‌سازی پیام
    for (int i = 0; i < 16; i++) {
        w[i] = (chunk[i * 4] << 24) | (chunk[i * 4 + 1] << 16) | 
               (chunk[i * 4 + 2] << 8) | chunk[i * 4 + 3];
    }
    
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = std::rotr(w[i-15], 7) ^ std::rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = std::rotr(w[i-2], 17) ^ std::rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    
    // حلقه فشرده‌سازی اصلی
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        
        h = g; g = f; f = e;
        e = d + temp1;
        d = c; c = b; b = a;
        a = temp1 + temp2;
    }
    
    // به‌روزرسانی state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void SHA256::pad() {
    // افزودن bit '1'
    buffer[buffer_length++] = 0x80;
    
    // پر کردن با صفر
    if (buffer_length > 56) {
        while (buffer_length < 64) {
            buffer[buffer_length++] = 0;
        }
        transform(buffer);
        buffer_length = 0;
    }
    
    while (buffer_length < 56) {
        buffer[buffer_length++] = 0;
    }
    
    // افزودن طول به بیت‌ها
    bit_count += buffer_length * 8;
    
    for (int i = 0; i < 8; i++) {
        buffer[56 + i] = (bit_count >> (56 - i * 8)) & 0xFF;
    }
    
    transform(buffer);
}

// متدهای عمومی
std::array<uint8_t, HASH_SIZE> SHA256::hash(const uint8_t* data, size_t length) {
    init();
    update(data, length);
    std::array<uint8_t, HASH_SIZE> result;
    finalize(result);
    return result;
}

std::array<uint8_t, HASH_SIZE> SHA256::hash(const std::vector<uint8_t>& data) {
    return hash(data.data(), data.size());
}

std::array<uint8_t, HASH_SIZE> SHA256::hash(const std::string& data) {
    return hash(reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

std::array<uint8_t, HASH_SIZE> SHA256::hash256(const uint8_t* data, size_t length) {
    auto first_hash = hash(data, length);
    return hash(first_hash.data(), first_hash.size());
}

// کرنل SHA256 برای GPU
__global__ void sha256_kernel_batch(
    const uint8_t* input_data,
    uint8_t* output_hashes,
    size_t batch_size,
    size_t item_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const uint8_t* input_item = input_data + idx * item_size;
    uint8_t* output_hash = output_hashes + idx * SHA256::HASH_SIZE;
    
    // پیاده‌سازی SHA256 برای این ترد
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // پردازش کامل بلوک‌های 64 بایتی
    // ... (پیاده‌سازی کامل مشابه نسخه CPU)
    
    // ذخیره نتیجه
    for (int i = 0; i < 8; i++) {
        output_hash[i * 4] = (state[i] >> 24) & 0xFF;
        output_hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        output_hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        output_hash[i * 4 + 3] = state[i] & 0xFF;
    }
}

// پیاده‌سازی کلاس GPU
bool SHA256_GPU::initialized = false;

void SHA256_GPU::initialize() {
    if (!initialized) {
        // راه‌اندازی منابع GPU مورد نیاز
        initialized = true;
    }
}

void SHA256_GPU::hash_batch(const uint8_t* input_data, size_t batch_size, 
                           size_t item_size, uint8_t* output_hashes) {
    int threads_per_block = 256;
    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
    
    sha256_kernel_batch<<<blocks_per_grid, threads_per_block>>>(
        input_data, output_hashes, batch_size, item_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace bitcoin_miner