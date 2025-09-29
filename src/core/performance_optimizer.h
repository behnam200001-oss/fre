#pragma once
#ifndef PERFORMANCE_OPTIMIZER_H
#define PERFORMANCE_OPTIMIZER_H

#include <cstdint>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>

namespace bitcoin_miner {

class PerformanceOptimizer {
public:
    struct OptimizationProfile {
        size_t optimal_batch_size;
        size_t threads_per_block;
        size_t blocks_per_grid;
        size_t shared_memory_size;
        bool use_pinned_memory;
        bool enable_concurrent_kernels;
        double estimated_performance;
    };
    
    PerformanceOptimizer();
    ~PerformanceOptimizer();
    
    // آنالیز و بهینه‌سازی
    OptimizationProfile analyze_gpu_performance(int device_id);
    OptimizationProfile find_optimal_settings(int device_id, size_t memory_available);
    
    // تنظیمات پویا
    void adjust_batch_size_based_on_memory();
    void optimize_memory_usage();
    void balance_workload(std::vector<int> device_ids);
    
    // مانیتورینگ
    void start_performance_monitoring();
    void stop_performance_monitoring();
    
    // آمار
    struct PerformanceStats {
        double average_utilization;
        double peak_utilization;
        size_t memory_usage;
        uint64_t keys_processed;
        double keys_per_second;
        uint32_t kernel_launches;
    };
    
    PerformanceStats get_performance_stats(int device_id) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // غیرقابل کپی
    PerformanceOptimizer(const PerformanceOptimizer&) = delete;
    PerformanceOptimizer& operator=(const PerformanceOptimizer&) = delete;
};

} // namespace bitcoin_miner

#endif // PERFORMANCE_OPTIMIZER_H