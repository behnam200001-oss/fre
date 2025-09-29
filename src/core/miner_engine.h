#pragma once
#ifndef ADVANCED_BITCOIN_MINER_ENGINE_H
#define ADVANCED_BITCOIN_MINER_ENGINE_H

#include <memory>
#include <atomic>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "../crypto/advanced_secp256k1.h"
#include "../bloom/super_bloom_filter.h"
#include "../address/address_factory.h"
#include "../gpu/multi_gpu_manager.h"
#include "../storage/result_manager.h"
#include "../monitoring/live_reporter.h"
#include "../config/config_manager.h"
#include "../utils/logger.h"
#include "../utils/exception_handler.h"

namespace bitcoin_miner {

class MinerEngine {
public:
    explicit MinerEngine();
    ~MinerEngine();

    // غیرقابل کپی
    MinerEngine(const MinerEngine&) = delete;
    MinerEngine& operator=(const MinerEngine&) = delete;

    // مدیریت چرخه حیات
    bool initialize(const std::string& config_file = "config/miner.conf");
    void start_mining();
    void stop_mining();
    void pause_mining();
    void resume_mining();
    
    // آمار real-time
    struct LiveStatistics {
        uint64_t total_keys_processed;
        uint64_t keys_per_second;
        uint32_t valid_matches;
        uint32_t false_positives;
        uint32_t gpu_count;
        double overall_utilization;
        std::chrono::seconds uptime;
        std::vector<std::string> recent_finds;
        double current_fp_rate;
    };

    LiveStatistics get_live_statistics() const;
    
    // مدیریت وضعیت
    enum class MiningState {
        STOPPED,
        INITIALIZING,
        RUNNING,
        PAUSED,
        SHUTTING_DOWN,
        ERROR_STATE
    };
    
    MiningState get_state() const { return current_state.load(std::memory_order_acquire); }
    
    // callbacks برای گزارش‌گیری
    using ResultCallback = std::function<void(const MiningResult&)>;
    using StatusCallback = std::function<void(MiningState, const std::string&)>;
    
    void set_result_callback(ResultCallback callback);
    void set_status_callback(StatusCallback callback);
    
    // متدهای جدید برای مدیریت Bloom Filter
    void reload_bloom_filter();
    double get_current_fp_rate() const;
    size_t get_bloom_filter_size() const;

private:
    // اعضای اصلی
    std::unique_ptr<MultiGPUManager> gpu_manager;
    std::unique_ptr<SuperBloomFilter> bloom_filter;
    std::unique_ptr<AddressFactory> address_factory;
    std::unique_ptr<ResultManager> result_manager;
    std::unique_ptr<LiveReporter> live_reporter;
    std::unique_ptr<ConfigManager> config_manager;
    
    // مدیریت وضعیت
    std::atomic<MiningState> current_state{MiningState::STOPPED};
    std::atomic<bool> should_stop{false};
    std::atomic<bool> is_paused{false};
    
    // آمار real-time
    std::atomic<uint64_t> total_keys_processed{0};
    std::atomic<uint32_t> valid_matches_found{0};
    std::atomic<uint32_t> false_positives_detected{0};
    std::atomic<uint64_t> start_time{0};
    std::atomic<double> current_fp_rate{0.0};
    
    // تردهای مدیریتی
    std::thread mining_thread;
    std::thread monitoring_thread;
    std::thread fp_monitor_thread;
    mutable std::mutex state_mutex;
    std::condition_variable pause_condition;
    
    // callbacks
    ResultCallback result_callback;
    StatusCallback status_callback;
    
    // اعلان توابع از دست رفته - اصلاح بحرانی 1
    bool load_configuration();
    void validate_environment() const;
    void setup_bloom_filter();
    void initialize_gpu_system();
    void cleanup_resources();
    void initialize_subsystems();
    
    // متدهای داخلی
    void mining_worker();
    void monitoring_worker();
    void fp_monitoring_worker();
    void update_state(MiningState new_state, const std::string& message = "");
    void handle_mining_result(const MiningResult& result);
    void signal_handler(int signal);
    
    // utility methods
    uint64_t calculate_uptime() const;
    uint64_t calculate_keys_per_second() const;
    void update_fp_rate_statistics();
};

} // namespace bitcoin_miner

#endif // ADVANCED_BITCOIN_MINER_ENGINE_H