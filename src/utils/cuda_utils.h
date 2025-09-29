#pragma once
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <mutex>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__; \
            ss << " - " << cudaGetErrorString(err) << " (" << #call << ")"; \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

namespace bitcoin_miner {

// Memory Pool برای مدیریت بهینه حافظه GPU
class CUDAMemoryPool {
private:
    struct MemoryBlock {
        void* device_ptr;
        size_t size;
        bool allocated;
        std::string tag;
    };
    
    std::vector<MemoryBlock> memory_blocks;
    std::mutex memory_mutex;
    
public:
    CUDAMemoryPool() = default;
    ~CUDAMemoryPool() {
        cleanup();
    }
    
    template<typename T>
    T* allocate(size_t count, const std::string& tag = "") {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        size_t size = count * sizeof(T);
        
        // پیدا کردن block آزاد
        for (auto& block : memory_blocks) {
            if (!block.allocated && block.size >= size) {
                block.allocated = true;
                block.tag = tag;
                return static_cast<T*>(block.device_ptr);
            }
        }
        
        // تخصیص جدید
        void* new_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&new_ptr, size));
        
        memory_blocks.push_back({new_ptr, size, true, tag});
        
        Logger::debug("Allocated {} bytes for {}", size, tag);
        return static_cast<T*>(new_ptr);
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        for (auto& block : memory_blocks) {
            if (block.device_ptr == ptr && block.allocated) {
                block.allocated = false;
                Logger::debug("Deallocated {} bytes for {}", block.size, block.tag);
                return;
            }
        }
        
        Logger::warning("Attempt to deallocate unknown pointer: {}", ptr);
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        for (auto& block : memory_blocks) {
            if (block.device_ptr) {
                CUDA_CHECK(cudaFree(block.device_ptr));
                block.device_ptr = nullptr;
            }
        }
        memory_blocks.clear();
        
        Logger::debug("Memory pool cleaned up");
    }
    
    size_t get_allocated_size() const {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        size_t total = 0;
        for (const auto& block : memory_blocks) {
            if (block.allocated) {
                total += block.size;
            }
        }
        return total;
    }
    
    void print_statistics() const {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        size_t allocated = 0;
        size_t total = 0;
        
        for (const auto& block : memory_blocks) {
            total += block.size;
            if (block.allocated) {
                allocated += block.size;
            }
        }
        
        Logger::info("Memory Pool: {}/{} bytes allocated ({:.1f}%)", 
                    allocated, total, (total > 0 ? (allocated * 100.0 / total) : 0.0));
    }
};

// RAII wrapper برای حافظه GPU
template<typename T>
class CUDABuffer {
private:
    T* d_ptr;
    size_t count;
    std::string tag;
    
public:
    CUDABuffer(size_t count, const std::string& tag = "") 
        : count(count), tag(tag) {
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
        Logger::debug("Allocated CUDABuffer: {} elements for {}", count, tag);
    }
    
    ~CUDABuffer() {
        if (d_ptr) {
            CUDA_CHECK(cudaFree(d_ptr));
            Logger::debug("Freed CUDABuffer: {}", tag);
        }
    }
    
    // غیرقابل کپی
    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;
    
    // قابل انتقال
    CUDABuffer(CUDABuffer&& other) noexcept 
        : d_ptr(other.d_ptr), count(other.count), tag(std::move(other.tag)) {
        other.d_ptr = nullptr;
        other.count = 0;
    }
    
    CUDABuffer& operator=(CUDABuffer&& other) noexcept {
        if (this != &other) {
            if (d_ptr) {
                CUDA_CHECK(cudaFree(d_ptr));
            }
            d_ptr = other.d_ptr;
            count = other.count;
            tag = std::move(other.tag);
            other.d_ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }
    
    T* data() { return d_ptr; }
    const T* data() const { return d_ptr; }
    size_t size() const { return count; }
    size_t bytes() const { return count * sizeof(T); }
    
    void copy_to_device(const T* h_data) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_data, bytes(), cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* h_data) const {
        CUDA_CHECK(cudaMemcpy(h_data, d_ptr, bytes(), cudaMemcpyDeviceToHost));
    }
    
    void memset(int value = 0) {
        CUDA_CHECK(cudaMemset(d_ptr, value, bytes()));
    }
};

} // namespace bitcoin_miner

#endif // CUDA_UTILS_H
