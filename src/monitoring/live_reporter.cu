#include "live_reporter.h"
#include "../utils/logger.h"
#include "../utils/format_utils.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace bitcoin_miner {

class LiveReporter::Impl {
public:
    Impl(DataCallback callback) 
        : data_callback(callback)
        , reporting_interval(std::chrono::seconds(30))
        , console_output(true)
        , file_output(false)
        , web_dashboard(false)
        , running(false) {}
    
    ~Impl() {
        stop_reporting();
    }
    
    void start_reporting() {
        if (running) return;
        
        running = true;
        reporter_thread = std::thread(&Impl::reporting_loop, this);
        Logger::info("Live reporting started");
    }
    
    void stop_reporting() {
        if (!running) return;
        
        running = false;
        if (reporter_thread.joinable()) {
            reporter_thread.join();
        }
        Logger::info("Live reporting stopped");
    }
    
    void set_report_interval(std::chrono::seconds interval) {
        reporting_interval = interval;
    }
    
    void enable_console_output(bool enabled) {
        console_output = enabled;
    }
    
    void enable_file_output(bool enabled, const std::string& filename) {
        file_output = enabled;
        if (enabled && !filename.empty()) {
            log_file.open(filename, std::ios::app);
            if (!log_file.is_open()) {
                Logger::error("Failed to open log file: {}", filename);
                file_output = false;
            }
        }
    }
    
    void enable_web_dashboard(bool enabled, int port) {
        web_dashboard = enabled;
        web_port = port;
        // راه‌اندازی سرور وب در اینجا
    }
    
    void update_stats() {
        if (!running) return;
        
        auto data = data_callback();
        last_data = data;
        
        if (console_output) {
            print_advanced_console_report(data);
        }
        
        if (file_output && log_file.is_open()) {
            print_file_report(data);
        }
    }
    
    void record_event(const std::string& event_type, const std::string& details) {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back({event_type, details, std::chrono::system_clock::now()});
        
        // حفظ فقط آخرین 1000 رویداد
        if (events.size() > 1000) {
            events.pop_front();
        }
        
        Logger::info("Event recorded: {} - {}", event_type, details);
    }
    
private:
    DataCallback data_callback;
    std::chrono::seconds reporting_interval;
    std::thread reporter_thread;
    std::atomic<bool> running;
    
    // خروجی‌ها
    bool console_output;
    bool file_output;
    std::ofstream log_file;
    
    bool web_dashboard;
    int web_port;
    
    // ذخیره داده‌ها
    LiveData last_data;
    std::mutex data_mutex;
    
    // مدیریت رویدادها
    struct Event {
        std::string type;
        std::string details;
        std::chrono::system_clock::time_point timestamp;
    };
    std::deque<Event> events;
    std::mutex events_mutex;
    
    void reporting_loop() {
        auto last_report = std::chrono::steady_clock::now();
        
        while (running) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report);
            
            if (elapsed >= reporting_interval) {
                update_stats();
                last_report = now;
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    void print_advanced_console_report(const LiveData& data) {
        std::cout << "\n\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║               🚀 ADVANCED MINING DASHBOARD 🚀                ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ " << FormatUtils::color_cyan("Time: ") << std::left << std::setw(52) 
                  << FormatUtils::get_current_timestamp() << " ║\n";
        std::cout << "║ " << FormatUtils::color_green("Total Keys: ") << std::setw(45) 
                  << FormatUtils::format_large_number(data.total_keys_processed) << " ║\n";
        std::cout << "║ " << FormatUtils::color_yellow("Speed: ") << std::setw(48)
                  << FormatUtils::format_hashrate(data.keys_per_second) << " ║\n";
        std::cout << "║ " << FormatUtils::color_green("Valid Matches: ") << std::setw(42) 
                  << data.valid_matches << " ║\n";
        std::cout << "║ " << FormatUtils::color_red("False Positives: ") << std::setw(40) 
                  << data.false_positives << " ║\n";
        std::cout << "║ " << FormatUtils::color_blue("FP Rate: ") << std::setw(45)
                  << std::scientific << std::setprecision(2) << data.current_fp_rate << " ║\n";
        std::cout << "║ " << FormatUtils::color_magenta("GPU Utilization: ") << std::setw(38) 
                  << FormatUtils::format_percentage(data.overall_utilization) << " ║\n";
        std::cout << "║ " << FormatUtils::color_cyan("Uptime: ") << std::setw(47)
                  << FormatUtils::format_duration(data.uptime.count()) << " ║\n";
        
        // نمایش وضعیت Bloom Filter (اگر در دسترس باشد)
        if (data.current_fp_rate > 0.0) {
            std::cout << "║ " << FormatUtils::color_white("Bloom FP Rate: ") << std::setw(41)
                      << std::scientific << std::setprecision(2) << data.current_fp_rate << " ║\n";
        }
        
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        
        // نمایش هشدارها
        if (data.current_fp_rate > 0.00001) {
            std::cout << FormatUtils::color_red("⚠️  WARNING: High FP rate detected - consider optimizing Bloom filter\n");
        }
        
        if (data.overall_utilization < 50.0) {
            std::cout << FormatUtils::color_yellow("⚠️  WARNING: Low GPU utilization - check workload distribution\n");
        }
        
        if (data.keys_per_second < 1000000) {
            std::cout << FormatUtils::color_yellow("⚠️  WARNING: Low mining speed - check system configuration\n");
        }
        
        // نمایش آخرین یافته‌ها
        if (!data.recent_finds.empty()) {
            std::cout << "\n🎉 RECENT FINDS:\n";
            for (size_t i = 0; i < std::min(data.recent_finds.size(), size_t(3)); i++) {
                std::cout << "   " << (i + 1) << ". " << data.recent_finds[i] << "\n";
            }
        }
        
        // نمایش وضعیت سیستم
        std::cout << "\n📊 SYSTEM STATUS:\n";
        std::cout << "   GPUs Active: " << data.gpu_count << "\n";
        std::cout << "   Overall Health: " << (data.overall_utilization > 70.0 ? "✅ Excellent" : 
                                              data.overall_utilization > 40.0 ? "⚠️ Good" : "❌ Poor") << "\n";
        
        std::cout << std::endl;
    }
    
    void print_file_report(const LiveData& data) {
        std::stringstream ss;
        ss << "REPORT " << FormatUtils::get_current_timestamp() << " | "
           << "Keys: " << data.total_keys_processed << " | "
           << "Speed: " << data.keys_per_second << " keys/s | "
           << "Matches: " << data.valid_matches << " | "
           << "False: " << data.false_positives << " | "
           << "FP Rate: " << std::scientific << data.current_fp_rate << " | "
           << "GPU: " << FormatUtils::format_percentage(data.overall_utilization) << " | "
           << "Uptime: " << data.uptime.count() << "s";
        
        log_file << ss.str() << std::endl;
        log_file.flush();
    }
};

// پیاده‌سازی متدهای اصلی LiveReporter
LiveReporter::LiveReporter(DataCallback callback) : impl(new Impl(callback)) {}
LiveReporter::~LiveReporter() = default;

void LiveReporter::start_reporting() { impl->start_reporting(); }
void LiveReporter::stop_reporting() { impl->stop_reporting(); }

void LiveReporter::set_report_interval(std::chrono::seconds interval) {
    impl->set_report_interval(interval);
}

void LiveReporter::enable_console_output(bool enabled) {
    impl->enable_console_output(enabled);
}

void LiveReporter::enable_file_output(bool enabled, const std::string& filename) {
    impl->enable_file_output(enabled, filename);
}

void LiveReporter::enable_web_dashboard(bool enabled, int port) {
    impl->enable_web_dashboard(enabled, port);
}

void LiveReporter::record_event(const std::string& event_type, const std::string& details) {
    impl->record_event(event_type, details);
}

void LiveReporter::update_stats() {
    impl->update_stats();
}

} // namespace bitcoin_miner