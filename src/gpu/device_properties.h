#pragma once
#ifndef DEVICE_PROPERTIES_H
#define DEVICE_PROPERTIES_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace bitcoin_miner {

struct DeviceProperties {
    int device_id;
    std::string name;
    int major;
    int minor;
    size_t total_global_mem;
    size_t shared_mem_per_block;
    int regs_per_block;
    int warp_size;
    size_t mem_pitch;
    int max_threads_per_block;
    int max_threads_dim[3];
    int max_grid_size[3];
    int clock_rate;
    size_t total_const_mem;
    int compute_capability_major;
    int compute_capability_minor;
    size_t texture_alignment;
    int concurrent_kernels;
    int ecc_enabled;
    int pci_bus_id;
    int pci_device_id;
    int tcc_driver;
    int unified_addressing;
    int memory_clock_rate;
    int memory_bus_width;
    int l2_cache_size;
    int max_threads_per_multi_processor;
    int compute_mode;
    int clock_memory;
    int global_memory_bus_width;
    int multi_processor_count;
    int max_shared_memory_per_multi_processor;
    int max_registers_per_multi_processor;
    int managed_memory;
    int is_multi_gpu_board;
    int host_native_atomic_supported;
    int single_to_double_precision_perf_ratio;
    
    DeviceProperties();
    
    // محاسبات کاربردی
    double get_performance_score() const;
    size_t get_max_recommended_batch_size() const;
    size_t get_optimal_threads_per_block() const;
    size_t get_optimal_blocks_per_grid(size_t problem_size) const;
    bool supports_concurrent_kernels() const;
    bool has_sufficient_memory(size_t required_memory) const;
    std::string get_compute_capability() const;
    
    // اعتبارسنجی
    bool validate() const;
    std::string to_string() const;
};

class DeviceManager {
public:
    DeviceManager();
    ~DeviceManager();
    
    // غیرقابل کپی
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    
    // مدیریت دستگاه‌ها
    static int get_device_count();
    static DeviceProperties get_device_properties(int device_id);
    static std::vector<DeviceProperties> get_all_devices();
    
    // پرس و جوهای پیشرفته
    static bool is_device_available(int device_id);
    static bool set_device(int device_id);
    static int get_current_device();
    static void reset_device(int device_id);
    
    // بهینه‌سازی
    static std::vector<int> get_optimal_device_order();
    static DeviceProperties get_best_device();
    static bool devices_have_same_architecture(const std::vector<int>& device_ids);
    
    // مانیتورینگ
    static size_t get_device_free_memory(int device_id);
    static size_t get_device_total_memory(int device_id);
    static double get_device_utilization(int device_id);
    static double get_device_temperature(int device_id);
    static double get_device_power_usage(int device_id);
    
    // اعتبارسنجی محیط
    static bool validate_environment();
    static std::vector<std::string> get_environment_issues();
    
    // لاگ اطلاعات
    static void log_device_info(int device_id);
    static void log_all_devices_info();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // متدهای کمکی
    static bool initialize_cuda();
    static void calculate_performance_scores();
};

// کلاس برای مدیریت چندین دستگاه
class MultiDeviceController {
public:
    MultiDeviceController();
    ~MultiDeviceController();
    
    // غیرقابل کپی
    MultiDeviceController(const MultiDeviceController&) = delete;
    MultiDeviceController& operator=(const MultiDeviceController&) = delete;
    
    // مدیریت دستگاه‌ها
    bool initialize(const std::vector<int>& device_ids);
    void cleanup();
    bool is_initialized() const;
    
    // توزیع کار
    std::vector<int> get_available_devices() const;
    int get_next_available_device();
    void mark_device_busy(int device_id);
    void mark_device_available(int device_id);
    
    // مانیتورینگ
    std::vector<DeviceProperties> get_device_status() const;
    double get_overall_utilization() const;
    size_t get_total_memory_used() const;
    size_t get_total_memory_available() const;
    
    // بهینه‌سازی
    void redistribute_workload();
    void adjust_device_priorities();
    bool detect_bottlenecks();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bitcoin_miner

#endif // DEVICE_PROPERTIES_H