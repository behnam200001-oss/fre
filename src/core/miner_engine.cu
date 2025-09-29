#include "miner_engine.h"
#include <csignal>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <secp256k1.h>
#include <secp256k1_recovery.h>

namespace bitcoin_miner {

// سیستم اعتبارسنجی دو مرحله‌ای بهبود یافته
class TwoStageValidator {
private:
    std::atomic<uint64_t> gpu_matches{0};
    std::atomic<uint64_t> cpu_validated_matches{0};
    std::atomic<uint64_t> false_positives{0};
    std::atomic<double> current_fp_rate{0.0};
    std::atomic<bool> needs_bloom_rebuild{false};
    
public:
    bool validate_match(const MiningResult& gpu_result) {
        if (!gpu_result.is_valid) {
            return false;
        }
        
        gpu_matches++;
        
        try {
            // مرحله 1: بررسی سریع با کتابخانه secp256k1
            if (!validate_private_key_format(gpu_result.private_key_bytes)) {
                false_positives++;
                update_fp_rate();
                return false;
            }
            
            // مرحله 2: تولید مجدد آدرس و مقایسه
            bool final_valid = regenerate_and_compare_address(gpu_result);
            
            if (final_valid) {
                cpu_validated_matches++;
                Logger::info("🎉 Valid match confirmed: {}", gpu_result.address);
                return true;
            } else {
                false_positives++;
                update_fp_rate();
                Logger::warning("False positive detected: {}", gpu_result.address);
                
                // اگر نرخ FP بسیار بالا باشد، پیشنهاد rebuild Bloom
                if (current_fp_rate > 0.000002) { // 2e-6
                    needs_bloom_rebuild.store(true, std::memory_order_release);
                    Logger::warning("High FP rate detected: {:.6f}, consider rebuilding Bloom filter", 
                                  current_fp_rate.load());
                }
                return false;
            }
            
        } catch (const std::exception& e) {
            Logger::error("Validation error for {}: {}", gpu_result.address, e.what());
            false_positives++;
            update_fp_rate();
            return false;
        }
    }
    
    bool needs_rebuild() const {
        return needs_bloom_rebuild.load(std::memory_order_acquire);
    }
    
    void reset_rebuild_flag() {
        needs_bloom_rebuild.store(false, std::memory_order_release);
    }
    
    double get_false_positive_rate() const {
        return current_fp_rate.load(std::memory_order_acquire);
    }
    
    void get_stats(uint64_t& gpu_m, uint64_t& cpu_m, uint64_t& fp) const {
        gpu_m = gpu_matches.load(std::memory_order_acquire);
        cpu_m = cpu_validated_matches.load(std::memory_order_acquire);
        fp = false_positives.load(std::memory_order_acquire);
    }
    
    void reset_stats() {
        gpu_matches = 0;
        cpu_validated_matches = 0;
        false_positives = 0;
        current_fp_rate = 0.0;
        needs_bloom_rebuild = false;
    }

private:
    bool validate_private_key_format(const std::vector<uint8_t>& private_key) {
        if (private_key.size() != 32) return false;
        
        // بررسی اینکه private key در محدوده معتبر باشد
        // مقایسه با n (حداکثر مقدار منحنی)
        uint32_t SECP256K1_N[8] = {
            0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6, 
            0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        };
        
        for (int i = 7; i >= 0; i--) {
            uint32_t key_part = (private_key[i * 4] << 24) | 
                               (private_key[i * 4 + 1] << 16) | 
                               (private_key[i * 4 + 2] << 8) | 
                               private_key[i * 4 + 3];
            
            if (key_part < SECP256K1_N[i]) return true;
            if (key_part > SECP256K1_N[i]) return false;
        }
        
        return false; // مساوی با n (نامعتبر)
    }
    
    bool regenerate_and_compare_address(const MiningResult& result) {
        // استفاده از کتابخانه secp256k1 برای تولید مجدد آدرس
        secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY | SECP256K1_CONTEXT_SIGN);
        if (!ctx) return false;
        
        bool valid = false;
        
        try {
            // بررسی معتبر بودن کلید خصوصی
            if (secp256k1_ec_seckey_verify(ctx, result.private_key_bytes.data()) != 1) {
                secp256k1_context_destroy(ctx);
                return false;
            }
            
            // تولید کلید عمومی از کلید خصوصی
            secp256k1_pubkey pubkey;
            if (secp256k1_ec_pubkey_create(ctx, &pubkey, result.private_key_bytes.data()) != 1) {
                secp256k1_context_destroy(ctx);
                return false;
            }
            
            // سریالایز کردن کلید عمومی (compressed)
            uint8_t pubkey_serialized[33];
            size_t output_len = 33;
            if (secp256k1_ec_pubkey_serialize(ctx, pubkey_serialized, &output_len, &pubkey, SECP256K1_EC_COMPRESSED) != 1) {
                secp256k1_context_destroy(ctx);
                return false;
            }
            
            // تولید آدرس از کلید عمومی (SHA256 + RIPEMD160 + Base58)
            std::string regenerated_address = generate_bitcoin_address(pubkey_serialized, 33, 0x00); // mainnet
            
            // مقایسه با آدرس یافت شده
            valid = (regenerated_address == result.address);
            
        } catch (const std::exception& e) {
            Logger::error("Address regeneration error: {}", e.what());
            valid = false;
        }
        
        secp256k1_context_destroy(ctx);
        return valid;
    }
    
    std::string generate_bitcoin_address(const uint8_t* public_key, size_t key_len, uint8_t version) {
        // SHA256 روی کلید عمومی
        SHA256 sha256;
        auto sha_hash = sha256.hash(public_key, key_len);
        
        // RIPEMD160 روی نتیجه SHA256
        RIPEMD160 ripemd160;
        auto ripemd_hash = ripemd160.hash(sha_hash);
        
        // اضافه کردن version byte
        std::vector<uint8_t> payload;
        payload.push_back(version); // 0x00 برای mainnet
        payload.insert(payload.end(), ripemd_hash.begin(), ripemd_hash.end());
        
        // محاسبه checksum (double SHA256)
        auto checksum_full = sha256.hash256(payload);
        std::vector<uint8_t> checksum(checksum_full.begin(), checksum_full.begin() + 4);
        
        // ترکیب payload و checksum
        payload.insert(payload.end(), checksum.begin(), checksum.end());
        
        // کدگذاری Base58
        return Base58::encode(payload);
    }
    
    void update_fp_rate() {
        uint64_t gpu_m = gpu_matches.load(std::memory_order_relaxed);
        uint64_t fp = false_positives.load(std::memory_order_relaxed);
        
        if (gpu_m > 1000) { // فقط پس از نمونه‌های کافی محاسبه شود
            double rate = static_cast<double>(fp) / gpu_m;
            current_fp_rate.store(rate, std::memory_order_relaxed);
        }
    }
};

// مدیریت سیگنال‌های سیستم
volatile std::sig_atomic_t g_signal_status = 0;

void signal_handler(int signal) {
    g_signal_status = signal;
}

MinerEngine::MinerEngine() {
    // نصب handler برای سیگنال‌های graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // راه‌اندازی logger
    Logger::initialize();
    Logger::info("MinerEngine initialized successfully");
}

MinerEngine::~MinerEngine() {
    stop_mining();
    cleanup_resources();
    Logger::info("MinerEngine shutdown complete");
}

bool MinerEngine::initialize(const std::string& config_file) {
    try {
        update_state(MiningState::INITIALIZING, "Starting initialization...");
        
        // بارگیری configuration
        if (!load_configuration()) {
            throw std::runtime_error("Failed to load configuration");
        }
        
        // اعتبارسنجی محیط اجرا
        validate_environment();
        
        // راه‌اندازی بلوم فیلتر بهینه‌شده
        setup_bloom_filter();
        
        // راه‌اندازی سیستم GPU
        initialize_gpu_system();
        
        // راه‌اندازی سیستم‌های دیگر
        initialize_subsystems();
        
        update_state(MiningState::STOPPED, "Initialization completed successfully");
        Logger::info("MinerEngine initialized successfully with optimized Bloom filter");
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Initialization failed: {}", e.what());
        update_state(MiningState::ERROR_STATE, e.what());
        return false;
    }
}

bool MinerEngine::load_configuration() {
    config_manager = std::make_unique<ConfigManager>();
    
    if (!config_manager->load_from_file("config/miner.conf")) {
        // استفاده از مقادیر پیش‌فرض اگر فایل پیکربندی وجود ندارد
        config_manager->set_defaults();
        Logger::warning("Using default configuration");
    }
    
    // اعتبارسنجی پیکربندی
    if (!config_manager->validate()) {
        throw std::runtime_error("Configuration validation failed");
    }
    
    Logger::info("Configuration loaded successfully");
    return true;
}

void MinerEngine::validate_environment() const {
    // بررسی وجود GPU
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA-capable GPU found");
    }
    
    // بررسی حافظه کافی
    size_t free_mem = 0;
    size_t total_mem = 0;
    cuda_status = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to query GPU memory");
    }
    
    size_t required_mem = 500 * 1024 * 1024; // 500MB حداقل
    
    if (free_mem < required_mem) {
        throw std::runtime_error("Insufficient GPU memory. Required: " + 
                               std::to_string(required_mem / (1024 * 1024)) + 
                               "MB, Available: " + 
                               std::to_string(free_mem / (1024 * 1024)) + "MB");
    }
    
    Logger::info("Environment validated: {} GPUs, {} MB free memory", 
                device_count, free_mem / (1024 * 1024));
}

void MinerEngine::setup_bloom_filter() {
    auto bloom_config = config_manager->get_bloom_config();
    bloom_filter = std::make_unique<SuperBloomFilter>(
        bloom_config.expected_elements,
        bloom_config.false_positive_rate,
        0x1234567890ABCDEFULL
    );
    
    if (!bloom_filter->initialize()) {
        throw std::runtime_error("Failed to initialize Bloom filter");
    }
    
    Logger::info("Bloom filter initialized: {} elements, {} MB", 
                bloom_config.expected_elements,
                bloom_filter->get_byte_size() / (1024 * 1024));
}

void MinerEngine::initialize_gpu_system() {
    gpu_manager = std::make_unique<MultiGPUManager>();
    
    if (!gpu_manager->initialize()) {
        throw std::runtime_error("Failed to initialize GPU system");
    }
    
    Logger::info("GPU system initialized successfully");
}

void MinerEngine::initialize_subsystems() {
    // راه‌اندازی سیستم گزارش‌گیری
    live_reporter = std::make_unique<LiveReporter>([this]() {
        LiveStatistics stats;
        stats.total_keys_processed = total_keys_processed.load(std::memory_order_acquire);
        stats.keys_per_second = calculate_keys_per_second();
        stats.valid_matches = valid_matches_found.load(std::memory_order_acquire);
        stats.false_positives = false_positives_detected.load(std::memory_order_acquire);
        stats.uptime = std::chrono::seconds(calculate_uptime());
        
        if (gpu_manager) {
            auto gpu_stats = gpu_manager->get_gpu_statistics();
            stats.gpu_count = gpu_stats.size();
            stats.overall_utilization = 0.0;
            
            for (const auto& gpu_stat : gpu_stats) {
                stats.overall_utilization += gpu_stat.utilization;
            }
            if (stats.gpu_count > 0) {
                stats.overall_utilization /= stats.gpu_count;
            }
        }
        
        stats.current_fp_rate = current_fp_rate.load(std::memory_order_acquire);
        return stats;
    });
    
    // راه‌اندازی سیستم ذخیره‌سازی نتایج
    result_manager = std::make_unique<ResultManager>();
    if (!result_manager->initialize("outputs/results")) {
        throw std::runtime_error("Failed to initialize result manager");
    }
    
    // راه‌اندازی سیستم آدرس
    address_factory = std::make_unique<AddressFactory>();
    
    // راه‌اندازی سیستم اعتبارسنجی
    validator = std::make_unique<TwoStageValidator>();
    
    Logger::info("All subsystems initialized successfully");
}

void MinerEngine::cleanup_resources() {
    // توقف همه تردها
    stop_mining();
    
    // پاک‌سازی منابع
    if (mining_thread.joinable()) {
        mining_thread.join();
    }
    if (monitoring_thread.joinable()) {
        monitoring_thread.join();
    }
    if (fp_monitor_thread.joinable()) {
        fp_monitor_thread.join();
    }
    
    // پاک‌سازی مدیران
    gpu_manager.reset();
    bloom_filter.reset();
    address_factory.reset();
    result_manager.reset();
    live_reporter.reset();
    config_manager.reset();
    validator.reset();
    
    Logger::info("All resources cleaned up");
}

void MinerEngine::start_mining() {
    MiningState current = current_state.load(std::memory_order_acquire);
    if (current == MiningState::RUNNING) {
        Logger::warning("Mining is already running");
        return;
    }
    
    try {
        update_state(MiningState::INITIALIZING, "Starting mining process...");
        
        // راه‌اندازی تردهای کاری
        should_stop.store(false, std::memory_order_release);
        is_paused.store(false, std::memory_order_release);
        start_time.store(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count(), 
            std::memory_order_release);
        
        mining_thread = std::thread(&MinerEngine::mining_worker, this);
        monitoring_thread = std::thread(&MinerEngine::monitoring_worker, this);
        fp_monitor_thread = std::thread(&MinerEngine::fp_monitoring_worker, this);
        
        update_state(MiningState::RUNNING, "Mining started successfully");
        Logger::info("Mining process started");
        
    } catch (const std::exception& e) {
        Logger::error("Failed to start mining: {}", e.what());
        update_state(MiningState::ERROR_STATE, e.what());
    }
}

void MinerEngine::stop_mining() {
    if (current_state.load(std::memory_order_acquire) == MiningState::STOPPED) {
        return;
    }
    
    update_state(MiningState::SHUTTING_DOWN, "Stopping mining process...");
    should_stop.store(true, std::memory_order_release);
    pause_condition.notify_all();
    
    if (mining_thread.joinable()) {
        mining_thread.join();
    }
    
    if (monitoring_thread.joinable()) {
        monitoring_thread.join();
    }
    
    if (fp_monitor_thread.joinable()) {
        fp_monitor_thread.join();
    }
    
    update_state(MiningState::STOPPED, "Mining stopped successfully");
    Logger::info("Mining process stopped");
}

void MinerEngine::pause_mining() {
    bool expected = false;
    if (is_paused.compare_exchange_weak(expected, true, std::memory_order_acq_rel)) {
        update_state(MiningState::PAUSED, "Mining paused");
        Logger::info("Mining paused");
    }
}

void MinerEngine::resume_mining() {
    bool expected = true;
    if (is_paused.compare_exchange_weak(expected, false, std::memory_order_acq_rel)) {
        pause_condition.notify_all();
        update_state(MiningState::RUNNING, "Mining resumed");
        Logger::info("Mining resumed");
    }
}

void MinerEngine::mining_worker() {
    Logger::info("Mining worker thread started");
    
    try {
        while (!should_stop.load(std::memory_order_acquire)) {
            // بررسی وضعیت pause با حافظه اتمی
            bool paused = is_paused.load(std::memory_order_acquire);
            if (paused) {
                std::unique_lock<std::mutex> lock(state_mutex);
                pause_condition.wait(lock, [this]() { 
                    return !is_paused.load(std::memory_order_acquire) || 
                           should_stop.load(std::memory_order_acquire); 
                });
                
                if (should_stop.load(std::memory_order_acquire)) break;
            }
            
            // اجرای دسته‌ای ماینینگ
            auto batch_results = gpu_manager->execute_mining_batch(1000000); // 1M keys per batch
            
            // پردازش نتایج
            for (const auto& result : batch_results) {
                handle_mining_result(result);
            }
            
            // بررسی سیگنال‌های سیستم
            if (g_signal_status != 0) {
                Logger::info("Received shutdown signal, stopping mining...");
                should_stop.store(true, std::memory_order_release);
                break;
            }
            
            // مدیریت هوشمند
            static auto last_adaptive_check = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::minutes>(now - last_adaptive_check).count() >= 5) {
                adaptive_bloom_management();
                dynamic_batch_management();
                last_adaptive_check = now;
            }
        }
    } catch (const std::exception& e) {
        Logger::error("Mining worker error: {}", e.what());
        update_state(MiningState::ERROR_STATE, e.what());
    }
    
    Logger::info("Mining worker thread finished");
}

void MinerEngine::monitoring_worker() {
    Logger::info("Monitoring worker thread started");
    
    while (!should_stop.load(std::memory_order_acquire)) {
        try {
            // آپدیت آمار real-time
            if (live_reporter) {
                live_reporter->update_stats();
            }
            
            // log آمار هر 30 ثانیه
            static auto last_log = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() >= 30) {
                auto stats = get_live_statistics();
                Logger::info("Stats - Keys: {}M, KPS: {}K, Matches: {}, FP Rate: {:.6f}, Uptime: {}h{}m{}s",
                           stats.total_keys_processed / 1000000,
                           stats.keys_per_second / 1000,
                           stats.valid_matches,
                           stats.current_fp_rate,
                           stats.uptime.count() / 3600,
                           (stats.uptime.count() % 3600) / 60,
                           stats.uptime.count() % 60);
                last_log = now;
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
        } catch (const std::exception& e) {
            Logger::error("Monitoring worker error: {}", e.what());
        }
    }
    
    Logger::info("Monitoring worker thread finished");
}

void MinerEngine::fp_monitoring_worker() {
    Logger::info("FP monitoring worker started");
    
    while (!should_stop.load(std::memory_order_acquire)) {
        try {
            update_fp_rate_statistics();
            std::this_thread::sleep_for(std::chrono::seconds(10));
        } catch (const std::exception& e) {
            Logger::error("FP monitoring error: {}", e.what());
        }
    }
    
    Logger::info("FP monitoring worker finished");
}

void MinerEngine::update_fp_rate_statistics() {
    uint64_t total_processed = total_keys_processed.load(std::memory_order_acquire);
    uint32_t false_positives = false_positives_detected.load(std::memory_order_acquire);
    
    if (total_processed > 0) {
        double fp_rate = static_cast<double>(false_positives) / total_processed;
        current_fp_rate.store(fp_rate, std::memory_order_release);
    }
}

// اضافه کردن مدیریت هوشمند Bloom Filter
void MinerEngine::adaptive_bloom_management() {
    if (!bloom_filter) return;
    
    double current_fp_rate = get_current_fp_rate();
    double bloom_saturation = bloom_filter->get_saturation();
    
    // اگر نرخ FP بسیار بالا یا اشباع Bloom بیش از 80% باشد
    if (current_fp_rate > 0.000005 || bloom_saturation > 0.8) { // 5e-6 یا 80% اشباع
        Logger::warning("Bloom filter optimization needed: FP rate {:.6f}, saturation {:.1f}%", 
                       current_fp_rate, bloom_saturation * 100);
        
        // بهینه‌سازی پارامترهای Bloom
        reload_bloom_filter();
        
        // ریست آمار validator
        if (validator) {
            validator->reset_rebuild_flag();
        }
    }
}

void MinerEngine::dynamic_batch_management() {
    if (!gpu_manager) return;
    
    double current_fp_rate = get_current_fp_rate();
    auto stats = get_live_statistics();
    
    // تنظیم پویا batch size بر اساس نرخ FP
    size_t optimal_batch_size = 1000000; // پیش‌فرض
    
    if (current_fp_rate > 0.00001) { // اگر FP rate > 1e-5
        optimal_batch_size = 500000; // کاهش batch size برای کاهش بار CPU
    } else if (current_fp_rate < 0.000001 && stats.keys_per_second > 5000000) { // اگر FP rate < 1e-6 و عملکرد خوب
        optimal_batch_size = 2000000; // افزایش batch size برای عملکرد بهتر
    }
    
    // اعمال batch size بهینه (اگر تغییر کرده)
    static size_t last_batch_size = 0;
    if (optimal_batch_size != last_batch_size) {
        Logger::info("Adjusting batch size to {} (FP rate: {:.6f})", 
                    optimal_batch_size, current_fp_rate);
        last_batch_size = optimal_batch_size;
        
        // اینجا باید batch size را به GPUManager اطلاع دهید
        // (بستگی به پیاده‌سازی GPUManager دارد)
    }
}

void MinerEngine::handle_mining_result(const MiningResult& result) {
    try {
        total_keys_processed.fetch_add(1, std::memory_order_relaxed);
        
        if (result.is_valid) {
            // استفاده از سیستم اعتبارسنجی دو مرحله‌ای بهبود یافته
            if (validator && validator->validate_match(result)) {
                valid_matches_found.fetch_add(1, std::memory_order_relaxed);
                
                // ذخیره نتیجه
                if (result_manager) {
                    result_manager->save_result(result);
                }
                
                // فراخوانی callback
                if (result_callback) {
                    result_callback(result);
                }
                
                Logger::info("🎉 CONFIRMED MATCH! Address: {}, Private Key: {}", 
                            result.address, result.private_key_hex);
            } else {
                false_positives_detected.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            false_positives_detected.fetch_add(1, std::memory_order_relaxed);
        }
        
    } catch (const std::exception& e) {
        Logger::error("Error handling mining result: {}", e.what());
    }
}

MinerEngine::LiveStatistics MinerEngine::get_live_statistics() const {
    LiveStatistics stats;
    stats.total_keys_processed = total_keys_processed.load(std::memory_order_acquire);
    stats.keys_per_second = calculate_keys_per_second();
    stats.valid_matches = valid_matches_found.load(std::memory_order_acquire);
    stats.false_positives = false_positives_detected.load(std::memory_order_acquire);
    stats.uptime = std::chrono::seconds(calculate_uptime());
    stats.current_fp_rate = current_fp_rate.load(std::memory_order_acquire);
    
    if (gpu_manager) {
        auto gpu_stats = gpu_manager->get_gpu_statistics();
        stats.gpu_count = gpu_stats.size();
        stats.overall_utilization = 0.0;
        
        for (const auto& gpu_stat : gpu_stats) {
            stats.overall_utilization += gpu_stat.utilization;
        }
        if (stats.gpu_count > 0) {
            stats.overall_utilization /= stats.gpu_count;
        }
    }
    
    return stats;
}

// پیاده‌سازی متدهای کمکی
uint64_t MinerEngine::calculate_uptime() const {
    uint64_t start = start_time.load(std::memory_order_acquire);
    if (start == 0) return 0;
    
    auto current_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    return current_time - start;
}

uint64_t MinerEngine::calculate_keys_per_second() const {
    auto uptime = calculate_uptime();
    if (uptime == 0) return 0;
    
    return total_keys_processed.load(std::memory_order_acquire) / uptime;
}

void MinerEngine::update_state(MiningState new_state, const std::string& message) {
    std::lock_guard<std::mutex> lock(state_mutex);
    current_state.store(new_state, std::memory_order_release);
    
    if (status_callback) {
        status_callback(new_state, message);
    }
    
    if (!message.empty()) {
        Logger::info("State changed to {}: {}", static_cast<int>(new_state), message);
    }
}

// پیاده‌سازی سایر متدهای عمومی
void MinerEngine::set_result_callback(ResultCallback callback) {
    result_callback = callback;
}

void MinerEngine::set_status_callback(StatusCallback callback) {
    status_callback = callback;
}

void MinerEngine::reload_bloom_filter() {
    Logger::info("Reloading Bloom filter...");
    try {
        if (bloom_filter) {
            setup_bloom_filter();
            Logger::info("Bloom filter reloaded successfully");
        }
    } catch (const std::exception& e) {
        Logger::error("Failed to reload Bloom filter: {}", e.what());
    }
}

double MinerEngine::get_current_fp_rate() const {
    return current_fp_rate.load(std::memory_order_acquire);
}

size_t MinerEngine::get_bloom_filter_size() const {
    if (bloom_filter) {
        return bloom_filter->get_byte_size();
    }
    return 0;
}

} // namespace bitcoin_miner