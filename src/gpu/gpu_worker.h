#pragma once
#ifndef GPU_WORKER_H
#define GPU_WORKER_H

#include "../core/mining_result.h"
#include <cstdint>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace bitcoin_miner {

// پیش‌اعلان کلاس
class SuperBloomFilter;

class GPUWorker {
public:
    struct Config {
        int device_id;
        size_t batch_size;
        size_t threads_per_block;
        size_t max_concurrent_batches;
        bool use_pinned_memory;
        double workload_percentage;
    };
    
    struct Statistics {
        uint64_t keys_processed{0};
        uint64_t batches_completed{0};
        uint64_t matches_found{0};
        double average_kps{0.0};
        double gpu_utilization{0.0};
        size_t memory_used{0};
        uint64_t last_update{0};
        bool is_running{false};
    };
    
    GPUWorker(int device_id, size_t batch_size = 1000000);
    ~GPUWorker();
    
    // مدیریت چرخه حیات
    bool initialize();
    void start();
    void stop();
    void pause();
    void resume();
    
    // اجرای کار
    std::vector<MiningResult> process_batch(uint64_t start_nonce, size_t batch_size);
    void set_bloom_filter(const std::shared_ptr<SuperBloomFilter>& filter);
    
    // آمار و وضعیت
    Statistics get_statistics() const;
    bool is_healthy() const;
    void reset_statistics();
    
    // پیکربندی پویا
    void update_config(const Config& new_config);
    Config get_config() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // غیرقابل کپی
    GPUWorker(const GPUWorker&) = delete;
    GPUWorker& operator=(const GPUWorker&) = delete;
};

} // namespace bitcoin_miner

#endif // GPU_WORKER_H