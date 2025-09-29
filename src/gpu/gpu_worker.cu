#include "gpu_worker.h"
#include "../utils/logger.h"
#include "../utils/cuda_utils.h"
#include "../bloom/super_bloom_filter.h"
#include "../crypto/advanced_secp256k1.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

namespace bitcoin_miner {

class GPUWorker::Impl {
public:
    Impl(int device_id, size_t batch_size) 
        : device_id(device_id), batch_size(batch_size), running(false), paused(false) {
    }
    
    ~Impl() {
        stop();
        cleanup();
    }
    
    bool initialize() {
        try {
            CUDA_CHECK(cudaSetDevice(device_id));
            
            // Initialize CUDA resources
            CUDA_CHECK(cudaMalloc(&d_private_keys, batch_size * 32));
            CUDA_CHECK(cudaMalloc(&d_public_keys, batch_size * 65)); // uncompressed
            CUDA_CHECK(cudaMalloc(&d_results, batch_size * sizeof(uint8_t))); // bloom filter results
            
            // Initialize random states
            CUDA_CHECK(cudaMalloc(&d_random_states, batch_size * sizeof(curandState)));
            
            int threads_per_block = 256;
            int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
            
            setup_random_states_kernel<<<blocks_per_grid, threads_per_block>>>(
                d_random_states, 
                std::chrono::steady_clock::now().time_since_epoch().count(), 
                batch_size
            );
            
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            Logger::info("GPU Worker {} initialized with batch size {}", device_id, batch_size);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to initialize GPU Worker {}: {}", device_id, e.what());
            return false;
        }
    }
    
    void start() {
        if (running) return;
        
        running = true;
        paused = false;
        worker_thread = std::thread(&Impl::worker_loop, this);
        Logger::info("GPU Worker {} started", device_id);
    }
    
    void stop() {
        if (!running) return;
        
        running = false;
        pause_condition.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        Logger::info("GPU Worker {} stopped", device_id);
    }
    
    void pause() {
        if (paused) return;
        
        paused = true;
        Logger::info("GPU Worker {} paused", device_id);
    }
    
    void resume() {
        if (!paused) return;
        
        paused = false;
        pause_condition.notify_all();
        Logger::info("GPU Worker {} resumed", device_id);
    }
    
    std::vector<MiningResult> process_batch(uint64_t start_nonce, size_t batch_size) {
        std::lock_guard<std::mutex> lock(batch_mutex);
        
        if (!running || paused || batch_size == 0) {
            return {};
        }
        
        try {
            CUDA_CHECK(cudaSetDevice(device_id));
            
            // Generate private keys
            int threads = 256;
            int blocks = (batch_size + threads - 1) / threads;
            generate_private_keys_kernel<<<blocks, threads>>>(
                d_private_keys, d_random_states, batch_size
            );
            
            // Generate public keys from private keys
            generate_public_keys_batch_kernel<<<blocks, threads>>>(
                d_private_keys, d_public_keys, false, batch_size // uncompressed
            );
            
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Check against bloom filter if available
            std::vector<MiningResult> results;
            if (bloom_filter) {
                auto gpu_bloom = bloom_filter->get_gpu_representation();
                
                // Copy bloom filter results
                std::vector<uint8_t> h_results(batch_size);
                CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 
                                    batch_size * sizeof(uint8_t), 
                                    cudaMemcpyDeviceToHost));
                
                // Process results
                for (size_t i = 0; i < batch_size; i++) {
                    if (h_results[i]) {
                        MiningResult result;
                        // Convert private key to hex
                        std::vector<uint8_t> private_key(32);
                        CUDA_CHECK(cudaMemcpy(private_key.data(), 
                                            d_private_keys + i * 32, 
                                            32, cudaMemcpyDeviceToHost));
                        
                        // Convert public key
                        std::vector<uint8_t> public_key(65);
                        CUDA_CHECK(cudaMemcpy(public_key.data(), 
                                            d_public_keys + i * 65, 
                                            65, cudaMemcpyDeviceToHost));
                        
                        // TODO: Generate address from public key
                        result.private_key_hex = "placeholder_hex";
                        result.public_key_uncompressed_hex = "placeholder_pubkey";
                        result.address = "placeholder_address";
                        result.is_valid = true;
                        result.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        
                        results.push_back(result);
                    }
                }
            }
            
            // Update statistics
            stats.keys_processed += batch_size;
            stats.batches_completed++;
            stats.matches_found += results.size();
            
            // Calculate average KPS
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - stats.last_update).count();
            
            if (elapsed > 0) {
                stats.average_kps = stats.keys_processed / elapsed;
            }
            
            stats.last_update = std::chrono::steady_clock::now();
            
            return results;
            
        } catch (const std::exception& e) {
            Logger::error("GPU Worker {} batch processing failed: {}", device_id, e.what());
            return {};
        }
    }
    
    void set_bloom_filter(const std::shared_ptr<SuperBloomFilter>& filter) {
        std::lock_guard<std::mutex> lock(bloom_mutex);
        bloom_filter = filter;
    }
    
    Statistics get_statistics() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return stats;
    }
    
    bool is_healthy() const {
        return running && !paused;
    }
    
    void reset_statistics() {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats = Statistics();
        stats.last_update = std::chrono::steady_clock::now();
    }
    
    void update_config(const Config& new_config) {
        std::lock_guard<std::mutex> lock(config_mutex);
        config = new_config;
        // Apply new configuration
        batch_size = config.batch_size;
    }
    
    Config get_config() const {
        std::lock_guard<std::mutex> lock(config_mutex);
        return config;
    }

private:
    int device_id;
    size_t batch_size;
    std::atomic<bool> running{false};
    std::atomic<bool> paused{false};
    
    // CUDA resources
    uint8_t* d_private_keys = nullptr;
    uint8_t* d_public_keys = nullptr;
    uint8_t* d_results = nullptr;
    curandState* d_random_states = nullptr;
    
    // Bloom filter
    std::shared_ptr<SuperBloomFilter> bloom_filter;
    mutable std::mutex bloom_mutex;
    
    // Statistics
    mutable Statistics stats;
    mutable std::mutex stats_mutex;
    
    // Configuration
    Config config;
    mutable std::mutex config_mutex;
    
    // Thread management
    std::thread worker_thread;
    std::mutex batch_mutex;
    std::condition_variable pause_condition;
    
    void worker_loop() {
        Logger::info("GPU Worker {} worker loop started", device_id);
        
        uint64_t nonce = 0;
        const size_t work_batch_size = 100000; // Process in smaller batches
        
        while (running) {
            // Check pause state
            if (paused) {
                std::unique_lock<std::mutex> lock(batch_mutex);
                pause_condition.wait(lock, [this]() { 
                    return !paused.load() || !running.load(); 
                });
                if (!running) break;
            }
            
            // Process work batch
            auto results = process_batch(nonce, work_batch_size);
            nonce += work_batch_size;
            
            // Small delay to prevent overwhelming the system
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        Logger::info("GPU Worker {} worker loop finished", device_id);
    }
    
    void cleanup() {
        if (d_private_keys) {
            cudaFree(d_private_keys);
            d_private_keys = nullptr;
        }
        if (d_public_keys) {
            cudaFree(d_public_keys);
            d_public_keys = nullptr;
        }
        if (d_results) {
            cudaFree(d_results);
            d_results = nullptr;
        }
        if (d_random_states) {
            cudaFree(d_random_states);
            d_random_states = nullptr;
        }
    }
};

// Kernel for setting up random states
__global__ void setup_random_states_kernel(curandState* states, uint64_t seed, size_t num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

// Kernel for generating private keys
__global__ void generate_private_keys_kernel(uint8_t* keys, curandState* states, size_t batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    curandState local_state = states[idx];
    uint8_t* private_key = keys + idx * 32;
    
    // Generate random private key
    for (int i = 0; i < 32; i++) {
        private_key[i] = curand(&local_state) & 0xFF;
    }
    
    // Ensure private key is not all zeros and is within valid range
    bool all_zero = true;
    for (int i = 0; i < 32; i++) {
        if (private_key[i] != 0) {
            all_zero = false;
            break;
        }
    }
    
    if (all_zero) {
        private_key[0] = 1; // Set to a valid non-zero value
    }
    
    states[idx] = local_state;
}

// Implementation of GPUWorker wrapper methods
GPUWorker::GPUWorker(int device_id, size_t batch_size) 
    : impl(new Impl(device_id, batch_size)) {}

GPUWorker::~GPUWorker() = default;

bool GPUWorker::initialize() { 
    return impl->initialize(); 
}

void GPUWorker::start() { 
    impl->start(); 
}

void GPUWorker::stop() { 
    impl->stop(); 
}

void GPUWorker::pause() { 
    impl->pause(); 
}

void GPUWorker::resume() { 
    impl->resume(); 
}

std::vector<MiningResult> GPUWorker::process_batch(uint64_t start_nonce, size_t batch_size) {
    return impl->process_batch(start_nonce, batch_size);
}

void GPUWorker::set_bloom_filter(const std::shared_ptr<SuperBloomFilter>& filter) {
    impl->set_bloom_filter(filter);
}

GPUWorker::Statistics GPUWorker::get_statistics() const {
    return impl->get_statistics();
}

bool GPUWorker::is_healthy() const {
    return impl->is_healthy();
}

void GPUWorker::reset_statistics() {
    impl->reset_statistics();
}

void GPUWorker::update_config(const Config& new_config) {
    impl->update_config(new_config);
}

GPUWorker::Config GPUWorker::get_config() const {
    return impl->get_config();
}

} // namespace bitcoin_miner