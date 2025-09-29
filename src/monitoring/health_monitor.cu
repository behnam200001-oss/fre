#include "health_monitor.h"
#include "../utils/logger.h"
#include "../utils/format_utils.h"
#include <fstream>
#include <sstream>
#ifdef __linux__
#include <sys/sysinfo.h>
#endif

namespace bitcoin_miner {

class HealthMonitor::Impl {
public:
    Impl() : monitoring(false), temperature_threshold(85.0), 
             memory_threshold(0.9), utilization_threshold(0.95) {}
    
    ~Impl() {
        stop_monitoring();
    }
    
    bool start_monitoring() {
        if (monitoring) return true;
        
        try {
            monitoring = true;
            monitoring_thread = std::thread(&Impl::monitoring_loop, this);
            Logger::info("Health monitoring started");
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to start health monitoring: {}", e.what());
            monitoring = false;
            return false;
        }
    }
    
    void stop_monitoring() {
        if (!monitoring) return;
        
        monitoring = false;
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
        Logger::info("Health monitoring stopped");
    }
    
    bool is_monitoring() const {
        return monitoring;
    }
    
    std::vector<GPUHealth> get_gpu_health() const {
        std::lock_guard<std::mutex> lock(health_mutex);
        return gpu_health;
    }
    
    SystemHealth get_system_health() const {
        std::lock_guard<std::mutex> lock(health_mutex);
        return system_health;
    }
    
    void set_temperature_threshold(double max_temp) {
        temperature_threshold = max_temp;
        Logger::info("Temperature threshold set to {}°C", max_temp);
    }
    
    void set_memory_threshold(double max_usage_percent) {
        memory_threshold = max_usage_percent / 100.0;
        Logger::info("Memory threshold set to {}%", max_usage_percent);
    }
    
    void set_utilization_threshold(double max_utilization) {
        utilization_threshold = max_utilization;
        Logger::info("Utilization threshold set to {}%", max_utilization * 100);
    }
    
    void set_alert_callback(AlertCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        alert_callback = callback;
    }
    
    void generate_health_report(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            Logger::error("Cannot create health report file: {}", filename);
            return;
        }
        
        auto gpu_health_data = get_gpu_health();
        auto system_health_data = get_system_health();
        
        file << "Advanced Bitcoin Miner - Health Report\n";
        file << "======================================\n";
        file << "Generated: " << FormatUtils::get_current_timestamp() << "\n\n";
        
        file << "SYSTEM HEALTH:\n";
        file << "  CPU Usage: " << std::fixed << std::setprecision(1) << system_health_data.cpu_usage << "%\n";
        file << "  Memory Usage: " << system_health_data.memory_used << " / " 
             << system_health_data.memory_total << " MB\n";
        file << "  Memory Usage: " << std::fixed << std::setprecision(1) 
             << (static_cast<double>(system_health_data.memory_used) / system_health_data.memory_total * 100) << "%\n";
        file << "  System Temperature: " << system_health_data.system_temperature << "°C\n";
        file << "  Uptime: " << FormatUtils::format_duration(system_health_data.uptime_seconds) << "\n";
        file << "  Overload Protection: " 
             << (system_health_data.overload_protection_active ? "ACTIVE" : "INACTIVE") << "\n\n";
        
        file << "GPU HEALTH:\n";
        for (const auto& gpu : gpu_health_data) {
            file << "  GPU " << gpu.device_id << " (" << gpu.name << "):\n";
            file << "    Utilization: " << std::fixed << std::setprecision(1) << gpu.utilization << "%\n";
            file << "    Temperature: " << gpu.temperature << "°C\n";
            file << "    Power Usage: " << gpu.power_usage << "W\n";
            file << "    Memory: " << gpu.memory_used << " / " << gpu.memory_total << " MB\n";
            file << "    Memory Usage: " << std::fixed << std::setprecision(1) 
                 << (static_cast<double>(gpu.memory_used) / gpu.memory_total * 100) << "%\n";
            file << "    Errors: " << gpu.errors << "\n";
            file << "    Health Status: " << (gpu.healthy ? "GOOD" : "POOR") << "\n";
            file << "    Status: " << (gpu.healthy ? "✅" : "❌") << "\n\n";
        }
        
        file << "ALERT SUMMARY:\n";
        file << "  Temperature Alerts: " << (check_temperature_alerts() ? "YES" : "NO") << "\n";
        file << "  Memory Alerts: " << (check_memory_alerts() ? "YES" : "NO") << "\n";
        file << "  Utilization Alerts: " << (check_utilization_alerts() ? "YES" : "NO") << "\n";
        
        file.close();
        Logger::info("Health report generated: {}", filename);
    }
    
    std::string get_health_summary() const {
        std::stringstream ss;
        
        auto gpu_health_data = get_gpu_health();
        auto system_health_data = get_system_health();
        
        ss << "Health Summary - " << FormatUtils::get_current_timestamp() << "\n";
        ss << "System: CPU=" << std::fixed << std::setprecision(1) << system_health_data.cpu_usage 
           << "%, Memory=" << (static_cast<double>(system_health_data.memory_used) / system_health_data.memory_total * 100) 
           << "%, Temp=" << system_health_data.system_temperature << "°C\n";
        
        ss << "GPUs: ";
        for (const auto& gpu : gpu_health_data) {
            ss << "GPU" << gpu.device_id << "(" << gpu.utilization << "%) ";
        }
        
        // Add alert status
        bool has_alerts = check_temperature_alerts() || check_memory_alerts() || check_utilization_alerts();
        if (has_alerts) {
            ss << " [ALERTS]";
        } else {
            ss << " [HEALTHY]";
        }
        
        return ss.str();
    }

private:
    std::atomic<bool> monitoring{false};
    std::thread monitoring_thread;
    
    mutable std::mutex health_mutex;
    std::vector<GPUHealth> gpu_health;
    SystemHealth system_health;
    
    mutable std::mutex callback_mutex;
    AlertCallback alert_callback;
    
    double temperature_threshold;
    double memory_threshold;
    double utilization_threshold;
    
    void monitoring_loop() {
        Logger::info("Health monitoring loop started");
        
        auto last_alert_check = std::chrono::steady_clock::now();
        
        while (monitoring) {
            try {
                update_gpu_health();
                update_system_health();
                
                // Check for alerts every 30 seconds
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_alert_check);
                
                if (elapsed.count() >= 30) {
                    check_alerts();
                    last_alert_check = now;
                }
                
                std::this_thread::sleep_for(std::chrono::seconds(5));
                
            } catch (const std::exception& e) {
                Logger::error("Health monitoring error: {}", e.what());
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
        }
        
        Logger::info("Health monitoring loop stopped");
    }
    
    void update_gpu_health() {
        std::vector<GPUHealth> new_health;
        
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        
        if (err != cudaSuccess || device_count == 0) {
            Logger::warning("No CUDA devices found for health monitoring");
            return;
        }
        
        for (int i = 0; i < device_count; i++) {
            GPUHealth health;
            health.device_id = i;
            
            try {
                cudaSetDevice(i);
                
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, i);
                health.name = props.name;
                
                // Get GPU utilization (placeholder - would use NVML in real implementation)
                health.utilization = 75.0 + (i * 5.0); // Placeholder
                
                // Get temperature (placeholder)
                health.temperature = 65.0 + (i * 3.0); // Placeholder
                
                // Get power usage (placeholder)
                health.power_usage = 150.0 + (i * 20.0); // Placeholder
                
                // Get memory usage
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                health.memory_used = (total_mem - free_mem) / (1024 * 1024);
                health.memory_total = total_mem / (1024 * 1024);
                
                health.errors = 0; // Placeholder
                health.healthy = check_gpu_health(health);
                
                new_health.push_back(health);
                
            } catch (const std::exception& e) {
                Logger::error("Failed to get health info for GPU {}: {}", i, e.what());
                health.healthy = false;
                new_health.push_back(health);
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(health_mutex);
            gpu_health = new_health;
        }
    }
    
    void update_system_health() {
        SystemHealth new_health;
        
        try {
            // CPU usage (placeholder - would use system-specific calls)
            new_health.cpu_usage = 30.0; // Placeholder
            
            // Memory usage
#ifdef __linux__
            struct sysinfo info;
            if (sysinfo(&info) == 0) {
                new_health.memory_total = (info.totalram * info.mem_unit) / (1024 * 1024);
                new_health.memory_used = ((info.totalram - info.freeram) * info.mem_unit) / (1024 * 1024);
            } else {
                new_health.memory_total = 16384; // 16GB placeholder
                new_health.memory_used = 8192;   // 8GB placeholder
            }
#else
            // Placeholder for other systems
            new_health.memory_total = 16384; // 16GB
            new_health.memory_used = 8192;   // 8GB
#endif
            
            // System temperature (placeholder)
            new_health.system_temperature = 45.0;
            
            // Uptime
#ifdef __linux__
            new_health.uptime_seconds = info.uptime;
#else
            new_health.uptime_seconds = 3600; // 1 hour placeholder
#endif
            
            // Overload protection
            new_health.overload_protection_active = check_system_overload(new_health);
            
        } catch (const std::exception& e) {
            Logger::error("Failed to update system health: {}", e.what());
        }
        
        {
            std::lock_guard<std::mutex> lock(health_mutex);
            system_health = new_health;
        }
    }
    
    bool check_gpu_health(const GPUHealth& health) const {
        // Check temperature
        if (health.temperature >= temperature_threshold) {
            return false;
        }
        
        // Check memory usage
        double memory_usage = static_cast<double>(health.memory_used) / health.memory_total;
        if (memory_usage >= memory_threshold) {
            return false;
        }
        
        // Check utilization
        if (health.utilization >= utilization_threshold * 100) {
            return false;
        }
        
        // Check for errors
        if (health.errors > 0) {
            return false;
        }
        
        return true;
    }
    
    bool check_system_overload(const SystemHealth& health) const {
        // Simple overload detection
        if (health.cpu_usage > 90.0) return true;
        
        double memory_usage = static_cast<double>(health.memory_used) / health.memory_total;
        if (memory_usage > 0.95) return true;
        
        if (health.system_temperature > 80.0) return true;
        
        return false;
    }
    
    void check_alerts() {
        auto current_gpu_health = get_gpu_health();
        auto current_system_health = get_system_health();
        
        // Check GPU alerts
        for (const auto& gpu : current_gpu_health) {
            if (gpu.temperature >= temperature_threshold) {
                trigger_alert("High temperature on GPU " + std::to_string(gpu.device_id) + 
                             ": " + std::to_string(gpu.temperature) + "°C", "WARNING");
            }
            
            double memory_usage = static_cast<double>(gpu.memory_used) / gpu.memory_total;
            if (memory_usage >= memory_threshold) {
                trigger_alert("High memory usage on GPU " + std::to_string(gpu.device_id) + 
                             ": " + std::to_string(memory_usage * 100) + "%", "WARNING");
            }
            
            if (gpu.utilization >= utilization_threshold * 100) {
                trigger_alert("High utilization on GPU " + std::to_string(gpu.device_id) + 
                             ": " + std::to_string(gpu.utilization) + "%", "WARNING");
            }
            
            if (gpu.errors > 0) {
                trigger_alert("GPU " + std::to_string(gpu.device_id) + " has " + 
                             std::to_string(gpu.errors) + " errors", "ERROR");
            }
        }
        
        // Check system alerts
        double system_memory_usage = static_cast<double>(current_system_health.memory_used) / 
                                   current_system_health.memory_total;
        if (system_memory_usage >= memory_threshold) {
            trigger_alert("High system memory usage: " + 
                         std::to_string(system_memory_usage * 100) + "%", "WARNING");
        }
        
        if (current_system_health.cpu_usage >= 90.0) {
            trigger_alert("High CPU usage: " + 
                         std::to_string(current_system_health.cpu_usage) + "%", "WARNING");
        }
        
        if (current_system_health.system_temperature >= temperature_threshold) {
            trigger_alert("High system temperature: " + 
                         std::to_string(current_system_health.system_temperature) + "°C", "WARNING");
        }
    }
    
    bool check_temperature_alerts() const {
        auto gpu_health_data = get_gpu_health();
        auto system_health_data = get_system_health();
        
        for (const auto& gpu : gpu_health_data) {
            if (gpu.temperature >= temperature_threshold) return true;
        }
        
        return system_health_data.system_temperature >= temperature_threshold;
    }
    
    bool check_memory_alerts() const {
        auto gpu_health_data = get_gpu_health();
        auto system_health_data = get_system_health();
        
        for (const auto& gpu : gpu_health_data) {
            double memory_usage = static_cast<double>(gpu.memory_used) / gpu.memory_total;
            if (memory_usage >= memory_threshold) return true;
        }
        
        double system_memory_usage = static_cast<double>(system_health_data.memory_used) / 
                                   system_health_data.memory_total;
        return system_memory_usage >= memory_threshold;
    }
    
    bool check_utilization_alerts() const {
        auto gpu_health_data = get_gpu_health();
        
        for (const auto& gpu : gpu_health_data) {
            if (gpu.utilization >= utilization_threshold * 100) return true;
        }
        
        return false;
    }
    
    void trigger_alert(const std::string& message, const std::string& severity) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (alert_callback) {
            alert_callback(message, severity);
        }
        Logger::warning("Health Alert [{}]: {}", severity, message);
    }
};

// Implementation of HealthMonitor wrapper methods
HealthMonitor::HealthMonitor() : impl(new Impl()) {}
HealthMonitor::~HealthMonitor() = default;

bool HealthMonitor::start_monitoring() { 
    return impl->start_monitoring(); 
}

void HealthMonitor::stop_monitoring() { 
    impl->stop_monitoring(); 
}

bool HealthMonitor::is_monitoring() const { 
    return impl->is_monitoring(); 
}

std::vector<HealthMonitor::GPUHealth> HealthMonitor::get_gpu_health() const {
    return impl->get_gpu_health();
}

HealthMonitor::SystemHealth HealthMonitor::get_system_health() const {
    return impl->get_system_health();
}

void HealthMonitor::set_temperature_threshold(double max_temp) {
    impl->set_temperature_threshold(max_temp);
}

void HealthMonitor::set_memory_threshold(double max_usage_percent) {
    impl->set_memory_threshold(max_usage_percent);
}

void HealthMonitor::set_utilization_threshold(double max_utilization) {
    impl->set_utilization_threshold(max_utilization);
}

void HealthMonitor::set_alert_callback(AlertCallback callback) {
    impl->set_alert_callback(callback);
}

void HealthMonitor::generate_health_report(const std::string& filename) const {
    impl->generate_health_report(filename);
}

std::string HealthMonitor::get_health_summary() const {
    return impl->get_health_summary();
}

} // namespace bitcoin_miner