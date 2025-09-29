#pragma once
#ifndef LIVE_REPORTER_H
#define LIVE_REPORTER_H

#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

namespace bitcoin_miner {

class LiveReporter {
public:
    struct LiveData {
        uint64_t total_keys_processed;
        uint64_t keys_per_second;
        uint32_t valid_matches;
        uint32_t false_positives;
        uint32_t gpu_count;
        double overall_utilization;
        std::chrono::seconds uptime;
        std::vector<std::string> recent_finds;
        std::string current_status;
    };
    
    using DataCallback = std::function<LiveData()>;
    
    LiveReporter(DataCallback callback);
    ~LiveReporter();
    
    // مدیریت گزارش‌گیری
    void start_reporting();
    void stop_reporting();
    void set_report_interval(std::chrono::seconds interval);
    
    // فرمت‌های خروجی
    void enable_console_output(bool enabled);
    void enable_file_output(bool enabled, const std::string& filename = "");
    void enable_web_dashboard(bool enabled, int port = 8080);
    
    // آمار پیشرفته
    void record_event(const std::string& event_type, const std::string& details);
    void update_stats();
    
    // هشدارها
    using AlertCallback = std::function<void(const std::string& alert, const std::string& severity)>;
    void set_alert_callback(AlertCallback callback);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // غیرقابل کپی
    LiveReporter(const LiveReporter&) = delete;
    LiveReporter& operator=(const LiveReporter&) = delete;
};

} // namespace bitcoin_miner

#endif // LIVE_REPORTER_H