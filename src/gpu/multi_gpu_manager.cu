#include "multi_gpu_manager.h"
#include "gpu_worker.h"
#include "../utils/logger.h"
#include "../utils/cuda_utils.h"
#include <algorithm>

namespace bitcoin_miner {

class MultiGPUManager::Impl {
public:
    Impl() : running(false) {}
    
    ~Impl() {
        stop_mining();
    }
    
    bool initialize() {
        try {
            int device_count = DeviceProperties::get_device_count();
            if (device_count == 0) {
                Logger::error("No CUDA-capable devices found");
                return false;
            }
            
            Logger::info("Found {} CUDA devices", device_count);
            
            // راه‌اندازی اطلاعات GPUها
            for (int i = 0; i < device_count; i++) {
                auto props = DeviceProperties::get_device_properties(i);
                GPUInfo info;
                info.device_id = i;
                info.name = props.name;
                info.compute_capability = props.major * 10 + props.minor;
                info.total_memory = props.totalGlobalMem;
                info.multiprocessor_count = props.multiProcessorCount;
                info.max_threads_per_block = props.maxThreadsPerBlock;
                info.enabled = true;
                info.performance_score = 1.0;
                
                gpu_info.push_back(info);
                
                Logger::info("GPU {}: {} (CC {}.{}, {} MB, {} SMs)", 
                            i, props.name, props.major, props.minor,
                            props.totalGlobalMem / (1024 * 1024),
                            props.multiProcessorCount);
            }
            
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("MultiGPUManager initialization failed: {}", e.what());
            return false;
        }
    }
    
    void start_mining() {
        if (running) {
            Logger::warning("Mining is already running");
            return;
        }
        
        running = true;
        workers.clear();
        
        // ایجاد worker برای هر GPU فعال
        for (const auto& gpu : gpu_info) {
            if (gpu.enabled) {
                auto worker = std::make_unique<GPUWorker>(gpu.device_id, 1000000);
                if (worker->initialize()) {
                    workers.push_back(std::move(worker));
                    workers.back()->start();
                    Logger::info("GPU worker {} started", gpu.device_id);
                } else {
                    Logger::error("Failed to initialize GPU worker {}", gpu.device_id);
                }
            }
        }
        
        Logger::info("Started mining on {} GPUs", workers.size());
    }
    
    void stop_mining() {
        if (!running) return;
        
        running = false;
        
        for (auto& worker : workers) {
            if (worker) {
                worker->stop();
            }
        }
        workers.clear();
        
        Logger::info("Stopped mining on all GPUs");
    }
    
    void pause_mining() {
        for (auto& worker : workers) {
            if (worker) {
                worker->pause();
            }
        }
        Logger::info("Mining paused on all GPUs");
    }
    
    void resume_mining() {
        for (auto& worker : workers) {
            if (worker) {
                worker->resume();
            }
        }
        Logger::info("Mining resumed on all GPUs");
    }
    
    std::vector<MiningResult> execute_mining_batch(size_t batch_size) {
        std::vector<MiningResult> results;
        
        if (!running || workers.empty()) {
            return results;
        }
        
        try {
            // توزیع کار بین GPUها
            size_t work_per_gpu = batch_size / workers.size();
            
            for (auto& worker : workers) {
                if (worker) {
                    auto batch_results = worker->process_batch(0, work_per_gpu);
                    results.insert(results.end(), batch_results.begin(), batch_results.end());
                }
            }
            
        } catch (const std::exception& e) {
            Logger::error("Batch execution failed: {}", e.what());
        }
        
        return results;
    }
    
    std::vector<GPUStatistics> get_gpu_statistics() const {
        std::vector<GPUStatistics> stats;
        
        for (const auto& worker : workers) {
            if (worker) {
                stats.push_back(worker->get_statistics());
            }
        }
        
        return stats;
    }
    
    size_t get_total_keys_processed() const {
        size_t total = 0;
        for (const auto& worker : workers) {
            if (worker) {
                total += worker->get_statistics().keys_processed;
            }
        }
        return total;
    }
    
    size_t get_total_matches_found() const {
        size_t total = 0;
        for (const auto& worker : workers) {
            if (worker) {
                total += worker->get_statistics().matches_found;
            }
        }
        return total;
    }
    
    double get_overall_utilization() const {
        if (workers.empty()) return 0.0;
        
        double total = 0.0;
        for (const auto& worker : workers) {
            if (worker) {
                total += worker->get_statistics().gpu_utilization;
            }
        }
        
        return total / workers.size();
    }
    
    void enable_gpu(int device_id, bool enabled) {
        for (auto& info : gpu_info) {
            if (info.device_id == device_id) {
                info.enabled = enabled;
                Logger::info("GPU {} {}", device_id, enabled ? "enabled" : "disabled");
                break;
            }
        }
    }
    
    void set_gpu_workload(int device_id, double percentage) {
        for (auto& info : gpu_info) {
            if (info.device_id == device_id) {
                info.performance_score = percentage / 100.0;
                Logger::info("GPU {} workload set to {}%", device_id, percentage);
                break;
            }
        }
    }
    
    std::vector<GPUInfo> get_gpu_info() const {
        return gpu_info;
    }
    
    void redistribute_workload() {
        Logger::info("Redistributing workload across GPUs");
        // در اینجا الگوریتم توزیع هوشمند بار پیاده‌سازی می‌شود
    }
    
    void adjust_batch_sizes() {
        Logger::info("Adjusting batch sizes based on performance");
        // تنظیم پویا اندازه batch بر اساس عملکرد
    }

private:
    std::vector<GPUInfo> gpu_info;
    std::vector<std::unique_ptr<GPUWorker>> workers;
    std::atomic<bool> running{false};
};

// پیاده‌سازی متدهای اصلی MultiGPUManager
MultiGPUManager::MultiGPUManager() : impl(new Impl()) {}
MultiGPUManager::~MultiGPUManager() = default;

bool MultiGPUManager::initialize() {
    return impl->initialize();
}

void MultiGPUManager::start_mining() {
    impl->start_mining();
}

void MultiGPUManager::stop_mining() {
    impl->stop_mining();
}

void MultiGPUManager::pause_mining() {
    impl->pause_mining();
}

void MultiGPUManager::resume_mining() {
    impl->resume_mining();
}

std::vector<MiningResult> MultiGPUManager::execute_mining_batch(size_t batch_size) {
    return impl->execute_mining_batch(batch_size);
}

std::vector<GPUStatistics> MultiGPUManager::get_gpu_statistics() const {
    return impl->get_gpu_statistics();
}

size_t MultiGPUManager::get_total_keys_processed() const {
    return impl->get_total_keys_processed();
}

size_t MultiGPUManager::get_total_matches_found() const {
    return impl->get_total_matches_found();
}

double MultiGPUManager::get_overall_utilization() const {
    return impl->get_overall_utilization();
}

void MultiGPUManager::enable_gpu(int device_id, bool enabled) {
    impl->enable_gpu(device_id, enabled);
}

void MultiGPUManager::set_gpu_workload(int device_id, double percentage) {
    impl->set_gpu_workload(device_id, percentage);
}

std::vector<GPUInfo> MultiGPUManager::get_gpu_info() const {
    return impl->get_gpu_info();
}

void MultiGPUManager::redistribute_workload() {
    impl->redistribute_workload();
}

void MultiGPUManager::adjust_batch_sizes() {
    impl->adjust_batch_sizes();
}

} // namespace bitcoin_miner