#pragma once
#ifndef ADVANCED_LOGGER_H
#define ADVANCED_LOGGER_H

#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <iostream>

namespace bitcoin_miner {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static void initialize(const std::string& log_file = "logs/miner.log", 
                          LogLevel level = LogLevel::INFO);
    static void shutdown();
    
    template<typename... Args>
    static void debug(const std::string& format, Args... args) {
        log(LogLevel::DEBUG, format, args...);
    }
    
    template<typename... Args>
    static void info(const std::string& format, Args... args) {
        log(LogLevel::INFO, format, args...);
    }
    
    template<typename... Args>
    static void warning(const std::string& format, Args... args) {
        log(LogLevel::WARNING, format, args...);
    }
    
    template<typename... Args>
    static void error(const std::string& format, Args... args) {
        log(LogLevel::ERROR, format, args...);
    }
    
    template<typename... Args>
    static void critical(const std::string& format, Args... args) {
        log(LogLevel::CRITICAL, format, args...);
    }
    
    static void set_log_level(LogLevel level);
    static LogLevel get_log_level();

private:
    template<typename... Args>
    static void log(LogLevel level, const std::string& format, Args... args) {
        if (level < current_level) return;
        
        std::lock_guard<std::mutex> lock(log_mutex);
        
        auto message = format_message(level, format, args...);
        write_to_console(level, message);
        write_to_file(message);
        
        if (level == LogLevel::CRITICAL) {
            flush();
        }
    }
    
    static std::string format_message(LogLevel level, const std::string& format);
    
    template<typename T, typename... Args>
    static std::string format_message(LogLevel level, const std::string& format, T value, Args... args) {
        auto pos = format.find("{}");
        if (pos == std::string::npos) {
            return format;
        }
        
        std::ostringstream oss;
        oss << value;
        std::string formatted = format.substr(0, pos) + oss.str() + format.substr(pos + 2);
        return format_message(level, formatted, args...);
    }
    
    static void write_to_console(LogLevel level, const std::string& message);
    static void write_to_file(const std::string& message);
    static void flush();
    
    static std::string get_timestamp();
    static std::string level_to_string(LogLevel level);
    static std::string get_color_code(LogLevel level);
    static std::string reset_color();
    
    static std::unique_ptr<std::ofstream> log_file;
    static std::mutex log_mutex;
    static LogLevel current_level;
    static bool initialized;
};

} // namespace bitcoin_miner

#endif // ADVANCED_LOGGER_H