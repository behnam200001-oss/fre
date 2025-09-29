#pragma once
#ifndef HEALTH_MONITOR_H
#define HEALTH_MONITOR_H

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>

namespace bitcoin_miner {

class HealthMonitor {
public:
    struct GPUHealth {
        int device_id;
        std::string name;
        double utilization;
        double temperature;
        double power_usage;
        size_t memory_used;
        size_t memory_total;
        uint64_t errors;
        bool healthy;
    };
    
    struct SystemHealth {
        double cpu_usage;
        size_t memory_used;
        size_t memory_total;
        double system_temperature;
        uint64_t uptime_seconds;
        bool overload_protection_active;
    };
    
    HealthMonitor();
    ~HealthMonitor();
    
    // مدیریت مانیتورینگ
    bool start_monitoring();
    void stop_monitoring();
    bool is_monitoring() const;
    
    // دریافت آمار سلامت
    std::vector<GPUHealth> get_gpu_health() const;
    SystemHealth get_system_health() const;
    
    // هشدارها و thresholds
    void set_temperature_threshold(double max_temp);
    void set_memory_threshold(double max_usage_percent);
    void set_utilization_threshold(double max_utilization);
    
    // callbacks برای هشدارها
    using AlertCallback = std::function<void(const std::string& alert_message, 
                                           const std::string& component)>;
    void set_alert_callback(AlertCallback callback);
    
    // گزارش‌گیری
    void generate_health_report(const std::string& filename) const;
    std::string get_health_summary() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // غیرقابل کپی
    HealthMonitor(const HealthMonitor&) = delete;
    HealthMonitor& operator=(const HealthMonitor&) = delete;
};

} // namespace bitcoin_miner

#endif // HEALTH_MONITOR_H