#pragma once
#ifndef LOAD_BALANCER_H
#define LOAD_BALANCER_H

#include <vector>
#include <memory>
#include <atomic>
#include <map>
#include <mutex>

namespace bitcoin_miner {

class LoadBalancer {
public:
    struct GPUWorkload {
        int device_id;
        double performance_score;
        size_t assigned_workload;
        size_t completed_workload;
        double utilization;
        bool is_available;
    };
    
    struct WorkChunk {
        uint64_t start_nonce;
        uint64_t end_nonce;
        int assigned_gpu;
        uint64_t priority;
        bool completed;
    };
    
    LoadBalancer();
    ~LoadBalancer();
    
    // مدیریت GPUها
    void register_gpu(int device_id, double performance_score = 1.0);
    void unregister_gpu(int device_id);
    void update_gpu_performance(int device_id, double performance_score);
    
    // توزیع کار
    WorkChunk get_next_chunk(int device_id, size_t preferred_size = 0);
    void mark_chunk_completed(const WorkChunk& chunk);
    void redistribute_workload();
    
    // آمار و مانیتورینگ
    std::vector<GPUWorkload> get_workload_distribution() const;
    double get_overall_efficiency() const;
    size_t get_pending_workload() const;
    
    // بهینه‌سازی پویا
    void adaptive_load_balancing();
    void detect_bottlenecks();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // غیرقابل کپی
    LoadBalancer(const LoadBalancer&) = delete;
    LoadBalancer& operator=(const LoadBalancer&) = delete;
};

} // namespace bitcoin_miner

#endif // LOAD_BALANCER_H