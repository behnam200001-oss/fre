#pragma once
#ifndef STATISTICS_MANAGER_H
#define STATISTICS_MANAGER_H

#include <cstdint>
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <mutex>

namespace bitcoin_miner {

class StatisticsManager {
public:
    struct TimeSeriesPoint {
        std::chrono::system_clock::time_point timestamp;
        double value;
    };
    
    struct GPUStats {
        int device_id;
        std::vector<TimeSeriesPoint> utilization;
        std::vector<TimeSeriesPoint> temperature;
        std::vector<TimeSeriesPoint> memory_usage;
        std::vector<TimeSeriesPoint> hashrate;
    };
    
    struct SystemStats {
        std::vector<TimeSeriesPoint> total_hashrate;
        std::vector<TimeSeriesPoint> matches_found;
        std::vector<TimeSeriesPoint> false_positives;
        std::vector<TimeSeriesPoint> efficiency;
    };
    
    StatisticsManager();
    ~StatisticsManager();
    
    // ثبت آمار
    void record_gpu_metric(int device_id, const std::string& metric, double value);
    void record_system_metric(const std::string& metric, double value);
    void record_match_found(const MiningResult& result);
    
    // بازیابی آمار
    std::vector<TimeSeriesPoint> get_metric_history(const std::string& metric, 
                                                   std::chrono::seconds duration) const;
    GPUStats get_gpu_stats(int device_id, std::chrono::seconds duration) const;
    SystemStats get_system_stats(std::chrono::seconds duration) const;
    
    // تجزیه و تحلیل
    double calculate_average(const std::string& metric, std::chrono::seconds duration) const;
    double calculate_peak(const std::string& metric, std::chrono::seconds duration) const;
    std::map<std::string, double> get_performance_metrics() const;
    
    // گزارش‌گیری
    std::string generate_summary_report() const;
    bool export_to_csv(const std::string& filename) const;
    bool export_to_json(const std::string& filename) const;
    
    // مدیریت داده
    void set_retention_period(std::chrono::hours period);
    void cleanup_old_data();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // غیرقابل کپی
    StatisticsManager(const StatisticsManager&) = delete;
    StatisticsManager& operator=(const StatisticsManager&) = delete;
};

} // namespace bitcoin_miner

#endif // STATISTICS_MANAGER_H