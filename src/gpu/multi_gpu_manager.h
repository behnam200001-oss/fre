#pragma once
#ifndef MULTI_GPU_MANAGER_H
#define MULTI_GPU_MANAGER_H

#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include "../core/mining_result.h"

namespace bitcoin_miner {

struct GPUInfo {
    int device_id;
    std::string name;
    int compute_capability;
    size_t total_memory;
    size_t multiprocessor_count;
    size_t max_threads_per_block;
    bool enabled;
    double performance_score;
};

struct GPUStatistics {
    int device_id;
    uint64_t keys_processed;
    uint64_t batches_completed;
    uint64_t matches_found;
    double average_kps;
    double utilization;
    double temperature;
    size_t memory_used;
    uint64_t last_update;
    bool is_healthy;
};

class GPUWorker;

class MultiGPUManager {
public:
    MultiGPUManager();
    ~MultiGPUManager();

    // غیرقابل کپی
    MultiGPUManager(const MultiGPUManager&) = delete;
    MultiGPUManager& operator=(const MultiGPUManager&) = delete;

    // مدیریت چرخه حیات
    bool initialize();
    void start_mining();
    void stop_mining();
    void pause_mining();
    void resume_mining();
    
    // اجرای دسته‌ای
    std::vector<MiningResult> execute_mining_batch(size_t batch_size);
    
    // آمار و مانیتورینگ
    std::vector<GPUStatistics> get_gpu_statistics() const;
    size_t get_total_keys_processed() const;
    size_t get_total_matches_found() const;
    double get_overall_utilization() const;
    
    // مدیریت GPUها
    void enable_gpu(int device_id, bool enabled);
    void set_gpu_workload(int device_id, double percentage);
    std::vector<GPUInfo> get_gpu_info() const;
    
    // بهینه‌سازی پویا
    void redistribute_workload();
    void adjust_batch_sizes();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bitcoin_miner

#endif // MULTI_GPU_MANAGER_H