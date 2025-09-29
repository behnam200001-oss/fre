#include "load_balancer.h"
#include "../utils/logger.h"
#include <algorithm>
#include <numeric>

namespace bitcoin_miner {

class LoadBalancer::Impl {
public:
    Impl() : next_chunk_id(1), total_work_assigned(0), total_work_completed(0) {}
    
    void register_gpu(int device_id, double performance_score) {
        std::lock_guard<std::mutex> lock(mutex);
        
        GPUWorkload workload;
        workload.device_id = device_id;
        workload.performance_score = performance_score;
        workload.assigned_workload = 0;
        workload.completed_workload = 0;
        workload.utilization = 0.0;
        workload.is_available = true;
        
        gpu_workloads[device_id] = workload;
        
        Logger::info("GPU {} registered with performance score {:.2f}", 
                    device_id, performance_score);
    }
    
    void unregister_gpu(int device_id) {
        std::lock_guard<std::mutex> lock(mutex);
        
        auto it = gpu_workloads.find(device_id);
        if (it != gpu_workloads.end()) {
            // Reassign work from this GPU to others
            redistribute_workload_internal();
            gpu_workloads.erase(it);
            Logger::info("GPU {} unregistered", device_id);
        }
    }
    
    void update_gpu_performance(int device_id, double performance_score) {
        std::lock_guard<std::mutex> lock(mutex);
        
        auto it = gpu_workloads.find(device_id);
        if (it != gpu_workloads.end()) {
            it->second.performance_score = performance_score;
            Logger::debug("GPU {} performance score updated to {:.2f}", 
                         device_id, performance_score);
            
            // Redistribute workload based on new performance scores
            redistribute_workload_internal();
        }
    }
    
    WorkChunk get_next_chunk(int device_id, size_t preferred_size) {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Check if GPU is registered and available
        auto it = gpu_workloads.find(device_id);
        if (it == gpu_workloads.end() || !it->second.is_available) {
            return WorkChunk{0, 0, -1, 0, true}; // Invalid chunk
        }
        
        size_t chunk_size = (preferred_size > 0) ? preferred_size : DEFAULT_CHUNK_SIZE;
        
        WorkChunk chunk;
        chunk.start_nonce = next_chunk_id * chunk_size;
        chunk.end_nonce = chunk.start_nonce + chunk_size;
        chunk.assigned_gpu = device_id;
        chunk.priority = next_chunk_id;
        chunk.completed = false;
        
        next_chunk_id++;
        
        // Update workload statistics
        it->second.assigned_workload += chunk_size;
        total_work_assigned += chunk_size;
        
        // Update utilization
        update_utilization();
        
        Logger::debug("Assigned chunk {}-{} to GPU {} (size: {})", 
                     chunk.start_nonce, chunk.end_nonce, device_id, chunk_size);
        
        return chunk;
    }
    
    void mark_chunk_completed(const WorkChunk& chunk) {
        std::lock_guard<std::mutex> lock(mutex);
        
        auto it = gpu_workloads.find(chunk.assigned_gpu);
        if (it != gpu_workloads.end()) {
            size_t chunk_size = chunk.end_nonce - chunk.start_nonce;
            it->second.completed_workload += chunk_size;
            total_work_completed += chunk_size;
            
            // Update utilization
            update_utilization();
        }
        
        Logger::debug("Chunk {}-{} completed by GPU {}", 
                     chunk.start_nonce, chunk.end_nonce, chunk.assigned_gpu);
    }
    
    void redistribute_workload() {
        std::lock_guard<std::mutex> lock(mutex);
        redistribute_workload_internal();
    }
    
    std::vector<GPUWorkload> get_workload_distribution() const {
        std::lock_guard<std::mutex> lock(mutex);
        
        std::vector<GPUWorkload> result;
        for (const auto& pair : gpu_workloads) {
            result.push_back(pair.second);
        }
        
        return result;
    }
    
    double get_overall_efficiency() const {
        std::lock_guard<std::mutex> lock(mutex);
        
        if (total_work_assigned == 0) return 0.0;
        return static_cast<double>(total_work_completed) / total_work_assigned;
    }
    
    size_t get_pending_workload() const {
        std::lock_guard<std::mutex> lock(mutex);
        return total_work_assigned - total_work_completed;
    }
    
    void adaptive_load_balancing() {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Calculate average utilization
        double avg_utilization = 0.0;
        int count = 0;
        
        for (const auto& pair : gpu_workloads) {
            if (pair.second.is_available) {
                avg_utilization += pair.second.utilization;
                count++;
            }
        }
        
        if (count > 0) {
            avg_utilization /= count;
        }
        
        // Detect imbalances and adjust
        const double threshold = 0.15; // 15% threshold
        
        for (auto& pair : gpu_workloads) {
            if (pair.second.is_available) {
                double deviation = std::abs(pair.second.utilization - avg_utilization);
                if (deviation > threshold) {
                    Logger::info("GPU {} utilization imbalance detected: {:.1f}% vs avg {:.1f}%", 
                               pair.first, pair.second.utilization * 100, avg_utilization * 100);
                }
            }
        }
        
        // Redistribute if significant imbalance detected
        if (avg_utilization > 0.0) {
            redistribute_workload_internal();
        }
    }
    
    void detect_bottlenecks() {
        std::lock_guard<std::mutex> lock(mutex);
        
        for (const auto& pair : gpu_workloads) {
            if (pair.second.is_available && pair.second.utilization < 0.3) {
                Logger::warning("Potential bottleneck on GPU {}: utilization only {:.1f}%", 
                               pair.first, pair.second.utilization * 100);
            }
        }
    }

private:
    static constexpr size_t DEFAULT_CHUNK_SIZE = 1000000; // 1M keys per chunk
    static constexpr double MIN_UTILIZATION = 0.1;
    
    mutable std::mutex mutex;
    std::map<int, GPUWorkload> gpu_workloads;
    uint64_t next_chunk_id;
    size_t total_work_assigned;
    size_t total_work_completed;
    
    void redistribute_workload_internal() {
        if (gpu_workloads.empty()) return;
        
        // Calculate total performance score
        double total_performance = 0.0;
        for (const auto& pair : gpu_workloads) {
            if (pair.second.is_available) {
                total_performance += pair.second.performance_score;
            }
        }
        
        if (total_performance <= 0.0) return;
        
        // Calculate target workload distribution
        Logger::info("Redistributing workload across {} GPUs", gpu_workloads.size());
        
        for (auto& pair : gpu_workloads) {
            if (pair.second.is_available) {
                double target_share = pair.second.performance_score / total_performance;
                Logger::debug("GPU {} target share: {:.1f}%", 
                            pair.first, target_share * 100);
            }
        }
    }
    
    void update_utilization() {
        for (auto& pair : gpu_workloads) {
            if (pair.second.assigned_workload > 0) {
                pair.second.utilization = static_cast<double>(pair.second.completed_workload) 
                                       / pair.second.assigned_workload;
            } else {
                pair.second.utilization = 0.0;
            }
            
            // Clamp utilization to [0, 1]
            pair.second.utilization = std::max(0.0, std::min(1.0, pair.second.utilization));
        }
    }
};

// Implementation of LoadBalancer wrapper methods
LoadBalancer::LoadBalancer() : impl(new Impl()) {}
LoadBalancer::~LoadBalancer() = default;

void LoadBalancer::register_gpu(int device_id, double performance_score) {
    impl->register_gpu(device_id, performance_score);
}

void LoadBalancer::unregister_gpu(int device_id) {
    impl->unregister_gpu(device_id);
}

void LoadBalancer::update_gpu_performance(int device_id, double performance_score) {
    impl->update_gpu_performance(device_id, performance_score);
}

LoadBalancer::WorkChunk LoadBalancer::get_next_chunk(int device_id, size_t preferred_size) {
    return impl->get_next_chunk(device_id, preferred_size);
}

void LoadBalancer::mark_chunk_completed(const WorkChunk& chunk) {
    impl->mark_chunk_completed(chunk);
}

void LoadBalancer::redistribute_workload() {
    impl->redistribute_workload();
}

std::vector<LoadBalancer::GPUWorkload> LoadBalancer::get_workload_distribution() const {
    return impl->get_workload_distribution();
}

double LoadBalancer::get_overall_efficiency() const {
    return impl->get_overall_efficiency();
}

size_t LoadBalancer::get_pending_workload() const {
    return impl->get_pending_workload();
}

void LoadBalancer::adaptive_load_balancing() {
    impl->adaptive_load_balancing();
}

void LoadBalancer::detect_bottlenecks() {
    impl->detect_bottlenecks();
}

} // namespace bitcoin_miner