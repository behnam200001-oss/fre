#pragma once
#ifndef EXCEPTION_HANDLER_H
#define EXCEPTION_HANDLER_H

#include <exception>
#include <string>
#include <sstream>
#include <system_error>

namespace bitcoin_miner {

class MinerException : public std::exception {
public:
    MinerException(const std::string& message, const std::string& component = "", int code = 0)
        : message_(message), component_(component), code_(code) {}
    
    const char* what() const noexcept override { return message_.c_str(); }
    const std::string& component() const { return component_; }
    int code() const { return code_; }
    std::string to_string() const;
    
private:
    std::string message_;
    std::string component_;
    int code_;
};

class GPUException : public MinerException {
public:
    GPUException(const std::string& message, int device_id = -1, int cuda_error = 0)
        : MinerException(message, "GPU", cuda_error), device_id_(device_id) {}
    
    int device_id() const { return device_id_; }
    
private:
    int device_id_;
};

class ConfigException : public MinerException {
public:
    ConfigException(const std::string& message, const std::string& config_key = "")
        : MinerException(message, "Configuration"), config_key_(config_key) {}
    
    const std::string& config_key() const { return config_key_; }
    
private:
    std::string config_key_;
};

class CryptoException : public MinerException {
public:
    CryptoException(const std::string& message, const std::string& algorithm = "")
        : MinerException(message, "Cryptography"), algorithm_(algorithm) {}
    
    const std::string& algorithm() const { return algorithm_; }
    
private:
    std::string algorithm_;
};

// هندلر استثناهای جهانی
class ExceptionHandler {
public:
    static void initialize();
    static void set_terminate_handler();
    static void set_signal_handlers();
    
    // هندلرهای خاص
    static void handle_cuda_error(cudaError_t error, const char* file, int line);
    static void handle_std_exception(const std::exception& e);
    static void handle_unknown_exception();
    
    // بازیابی و restore
    static bool should_restart_after_exception();
    static void cleanup_after_exception();
    
    // گزارش‌گیری
    static void generate_crash_report(const std::exception& e);
    static std::string get_last_exception_info();
};

// ماکروهای مفید
#define MINER_TRY try {
#define MINER_CATCH } catch (const bitcoin_miner::MinerException& e) { \
    ExceptionHandler::handle_std_exception(e); \
} catch (const std::exception& e) { \
    ExceptionHandler::handle_std_exception(e); \
} catch (...) { \
    ExceptionHandler::handle_unknown_exception(); \
}

#define CUDA_SAFE_CALL(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            ExceptionHandler::handle_cuda_error(err, __FILE__, __LINE__); \
        } \
    } while(0)

} // namespace bitcoin_miner

#endif // EXCEPTION_HANDLER_H