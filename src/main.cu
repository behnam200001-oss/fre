#include "core/miner_engine.h"
#include "utils/logger.h"
#include "utils/exception_handler.h"
#include "config/config_manager.h"

#include <iostream>
#include <csignal>
#include <chrono>
#include <thread>

using namespace bitcoin_miner;

// Handler برای graceful shutdown
volatile std::sig_atomic_t g_shutdown_requested = 0;

void signal_handler(int signal) {
    g_shutdown_requested = signal;
    std::cout << "\n📡 Received shutdown signal (" << signal << "), initiating graceful shutdown...\n";
}

void print_banner() {
    std::cout << R"(
╔════════════════════════════════════════════════════════════════╗
║               🚀 Advanced Bitcoin Miner CUDA 🚀               ║
║                  Optimized Edition v2.1.0                     ║
║       50M addresses • p=1e-6 FP • Double Hashing • 5M KPS     ║
╚════════════════════════════════════════════════════════════════╝
    )" << std::endl;
}

void print_optimized_stats() {
    std::cout << "🎯 Optimized Configuration:\n";
    std::cout << "   • Bloom Filter: 50M addresses, p=1e-6\n";
    std::cout << "   • Memory: ~200 MB, 20 hash functions\n";
    std::cout << "   • Target: 5,000,000 keys/second\n";
    std::cout << "   • Expected FP: ~18,000 per hour\n";
    std::cout << "   • Double Hashing: Enabled\n";
    std::cout << "   • Two-Stage Validation: Enabled\n";
    std::cout << "   • Memory Pool: Enabled\n";
    std::cout << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  -c, --config FILE    Configuration file (default: config/miner.conf)\n"
              << "  -d, --debug          Enable debug mode\n"
              << "  -p, --profile        Enable profiling mode\n"
              << "  --reload-bloom       Reload Bloom filter\n"
              << "  --fp-stats           Show false positive statistics\n"
              << "  -h, --help           Show this help message\n"
              << "  -v, --version        Show version information\n"
              << "\nExamples:\n"
              << "  " << program_name << " -c my_config.conf\n"
              << "  " << program_name << " --debug --fp-stats\n" 
              << std::endl;
}

void print_version() {
    std::cout << "Advanced Bitcoin Miner CUDA v2.1.0 (Optimized)\n"
              << "Build: " << __DATE__ << " " << __TIME__ << "\n"
              << "CUDA: " << CUDART_VERSION << "\n"
              << "Bloom Filter: 50M addresses, p=1e-6, Double Hashing\n"
              << "Supported GPU Architectures: sm_70, sm_75, sm_80, sm_86\n" 
              << std::endl;
}

int main(int argc, char* argv[]) {
    // تنظیم signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    print_banner();
    print_optimized_stats();
    
    // پارامترهای خط فرمان
    std::string config_file = "config/miner.conf";
    bool debug_mode = false;
    bool profile_mode = false;
    bool reload_bloom = false;
    bool show_fp_stats = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            print_version();
            return 0;
        } else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config_file = argv[++i];
            } else {
                std::cerr << "❌ Error: --config requires a file path" << std::endl;
                return 1;
            }
        } else if (arg == "-d" || arg == "--debug") {
            debug_mode = true;
        } else if (arg == "-p" || arg == "--profile") {
            profile_mode = true;
        } else if (arg == "--reload-bloom") {
            reload_bloom = true;
        } else if (arg == "--fp-stats") {
            show_fp_stats = true;
        } else {
            std::cerr << "❌ Error: Unknown option " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    try {
        // تنظیم سطح لاگ بر اساس حالت debug
        if (debug_mode) {
            Logger::set_log_level(LogLevel::DEBUG);
            Logger::info("Debug mode enabled");
        }
        
        if (profile_mode) {
            Logger::info("Profiling mode enabled");
        }
        
        Logger::info("Starting Advanced Bitcoin Miner CUDA v2.1.0 (Optimized)");
        Logger::info("Configuration file: {}", config_file);
        
        // ایجاد و راه‌اندازی موتور ماینینگ
        MinerEngine miner;
        
        Logger::info("Initializing miner engine with optimized parameters...");
        if (!miner.initialize(config_file)) {
            Logger::critical("Failed to initialize miner engine");
            return 1;
        }
        
        // reload Bloom filter اگر درخواست شده
        if (reload_bloom) {
            Logger::info("Reloading Bloom filter...");
            miner.reload_bloom_filter();
        }
        
        // تنظیم callbacks برای گزارش‌گیری
        miner.set_status_callback([](MinerEngine::MiningState state, const std::string& message) {
            Logger::info("Miner state changed: {} - {}", static_cast<int>(state), message);
        });
        
        miner.set_result_callback([](const MiningResult& result) {
            Logger::info("🎉 CONFIRMED MATCH! Address: {}, Private Key: {}", 
                        result.address, result.private_key_hex);
        });
        
        Logger::info("Starting mining process with optimized configuration...");
        miner.start_mining();
        
        // حلقه اصلی برای graceful shutdown
        Logger::info("Miner is running. Press Ctrl+C to stop...");
        
        while (!g_shutdown_requested) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // نمایش آمار هر 60 ثانیه
            static auto last_stats_time = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time);
            
            if (elapsed.count() >= 60) {
                auto stats = miner.get_live_statistics();
                Logger::info("Live Stats - Keys: {}M, KPS: {}K, Matches: {}, FP Rate: {:.6f}, Uptime: {}h{}m{}s",
                           stats.total_keys_processed / 1000000,
                           stats.keys_per_second / 1000,
                           stats.valid_matches,
                           stats.current_fp_rate,
                           stats.uptime.count() / 3600,
                           (stats.uptime.count() % 3600) / 60,
                           stats.uptime.count() % 60);
                last_stats_time = now;
            }
            
            // نمایش آمار FP اگر درخواست شده
            if (show_fp_stats) {
                static auto last_fp_time = std::chrono::steady_clock::now();
                auto fp_now = std::chrono::steady_clock::now();
                auto fp_elapsed = std::chrono::duration_cast<std::chrono::seconds>(fp_now - last_fp_time);
                
                if (fp_elapsed.count() >= 30) {
                    double fp_rate = miner.get_current_fp_rate();
                    size_t bloom_size = miner.get_bloom_filter_size();
                    
                    std::cout << "📊 FP Statistics - Rate: " << std::scientific << fp_rate 
                              << ", Bloom Size: " << bloom_size / (1024 * 1024) << " MB" << std::endl;
                    last_fp_time = fp_now;
                }
            }
        }
        
        // Graceful shutdown
        Logger::info("Shutting down miner engine...");
        miner.stop_mining();
        
        // نمایش آمار نهایی
        auto final_stats = miner.get_live_statistics();
        Logger::info("Final Statistics:");
        Logger::info("  Total Keys Processed: {}", final_stats.total_keys_processed);
        Logger::info("  Average Speed: {} keys/sec", final_stats.keys_per_second);
        Logger::info("  Valid Matches: {}", final_stats.valid_matches);
        Logger::info("  False Positives: {}", final_stats.false_positives);
        Logger::info("  Final FP Rate: {:.6f}", final_stats.current_fp_rate);
        Logger::info("  Total Uptime: {} seconds", final_stats.uptime.count());
        Logger::info("  Bloom Filter Size: {} MB", miner.get_bloom_filter_size() / (1024 * 1024));
        
        Logger::info("Miner shutdown completed successfully");
        std::cout << "✨ Miner stopped gracefully. Goodbye!" << std::endl;
        
    } catch (const std::exception& e) {
        Logger::critical("Fatal error: {}", e.what());
        std::cerr << "💥 Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}