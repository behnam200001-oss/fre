#include "random.h"
#include "logger.h"
#include <random>
#include <chrono>

namespace bitcoin_miner {

// تعریف اعضای استاتیک
std::mutex SecureRandom::mutex_;
std::random_device SecureRandom::rd_;
std::mt19937_64 SecureRandom::generator_;
std::atomic<bool> SecureRandom::seeded_(false);

SecureRandom::SecureRandom() {
    ensure_seeded();
}

SecureRandom::~SecureRandom() {
    // Cleanup if needed
}

void SecureRandom::ensure_seeded() {
    if (!seeded_.load()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!seeded_.load()) {
            // استفاده از چندین منبع برای entropy
            auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            auto thread_seed = std::hash<std::thread::id>{}(std::this_thread::get_id());
            
            uint64_t seed = rd_() ^ time_seed ^ thread_seed;
            generator_.seed(seed);
            seeded_ = true;
            
            Logger::debug("SecureRandom seeded successfully");
        }
    }
}

uint32_t SecureRandom::next_uint32() {
    ensure_seeded();
    std::lock_guard<std::mutex> lock(mutex_);
    return std::uniform_int_distribution<uint32_t>{}(generator_);
}

uint64_t SecureRandom::next_uint64() {
    ensure_seeded();
    std::lock_guard<std::mutex> lock(mutex_);
    return std::uniform_int_distribution<uint64_t>{}(generator_);
}

void SecureRandom::next_bytes(uint8_t* buffer, size_t length) {
    ensure_seeded();
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (size_t i = 0; i < length; ++i) {
        buffer[i] = static_cast<uint8_t>(std::uniform_int_distribution<int>{0, 255}(generator_));
    }
}

std::vector<uint8_t> SecureRandom::next_bytes(size_t length) {
    std::vector<uint8_t> result(length);
    next_bytes(result.data(), length);
    return result;
}

std::vector<uint8_t> SecureRandom::generate_private_key() {
    std::vector<uint8_t> private_key(32);
    
    // تولید کلید خصوصی با توزیع یکنواخت
    next_bytes(private_key.data(), 32);
    
    // اطمینان از معتبر بودن کلید (کمتر از n باشد)
    // این بررسی در سطح بالاتر انجام می‌شود
    
    return private_key;
}

void SecureRandom::seed(uint64_t seed_value) {
    std::lock_guard<std::mutex> lock(mutex_);
    generator_.seed(seed_value);
    seeded_ = true;
}

void SecureRandom::seed_from_system() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto seed = rd_() ^ std::chrono::high_resolution_clock::now().time_since_epoch().count();
    generator_.seed(seed);
    seeded_ = true;
}

bool SecureRandom::has_sufficient_entropy() {
    return seeded_.load() && rd_.entropy() > 0;
}

// پیاده‌سازی نسخه GPU
bool GPURandom::initialized_ = false;

void GPURandom::initialize() {
    if (!initialized_) {
        // راه‌اندازی منابع GPU مورد نیاز
        // می‌تواند شامل تخصیص حافظه برای حالت‌های تصادفی باشد
        initialized_ = true;
        Logger::debug("GPURandom initialized");
    }
}

void GPURandom::cleanup() {
    if (initialized_) {
        // پاک‌سازی منابع GPU
        initialized_ = false;
        Logger::debug("GPURandom cleaned up");
    }
}

__global__ void generate_private_keys_kernel(
    uint8_t* keys_buffer,
    curandState* random_states,
    size_t batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    curandState local_state = random_states[idx];
    uint8_t* private_key = keys_buffer + idx * 32;
    
    // تولید کلید خصوصی با استفاده از curand
    for (int i = 0; i < 32; i++) {
        private_key[i] = curand(&local_state) & 0xFF;
    }
    
    // بررسی اعتبار کلید (اساسی)
    bool valid = true;
    for (int i = 0; i < 32; i++) {
        if (private_key[i] != 0) {
            valid = true;
            break;
        }
    }
    
    // اگر کلید نامعتبر بود، دوباره تولید کن
    if (!valid) {
        for (int i = 0; i < 32; i++) {
            private_key[i] = curand(&local_state) & 0xFF;
        }
    }
    
    random_states[idx] = local_state;
}

void GPURandom::generate_private_keys_batch(uint8_t* keys_buffer, size_t batch_size) {
    if (!initialized_) {
        initialize();
    }
    
    // تخصیص حافظه برای حالت‌های تصادفی
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, batch_size * sizeof(curandState)));
    
    // راه‌اندازی حالت‌های تصادفی
    setup_random_states_kernel<<<(batch_size + 255) / 256, 256>>>(
        d_states, 
        std::chrono::steady_clock::now().time_since_epoch().count(), 
        batch_size
    );
    
    // اجرای کرنل تولید کلیدها
    int threads_per_block = 256;
    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
    
    generate_private_keys_kernel<<<blocks_per_grid, threads_per_block>>>(
        keys_buffer, d_states, batch_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // پاک‌سازی
    CUDA_CHECK(cudaFree(d_states));
}

void GPURandom::setup_random_states(void** states_ptr, size_t num_states, uint64_t seed) {
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(states_ptr, num_states * sizeof(curandState)));
    d_states = static_cast<curandState*>(*states_ptr);
    
    // راه‌اندازی حالت‌های تصادفی
    int threads_per_block = 256;
    int blocks_per_grid = (num_states + threads_per_block - 1) / threads_per_block;
    
    setup_random_states_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_states, seed, num_states
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// کرنل برای راه‌اندازی حالت‌های تصادفی
__global__ void setup_random_states_kernel(curandState* states, uint64_t seed, size_t num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

} // namespace bitcoin_miner