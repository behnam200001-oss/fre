#pragma once
#ifndef FORMAT_UTILS_H
#define FORMAT_UTILS_H

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace bitcoin_miner {

class FormatUtils {
public:
    // فرمت اعداد بزرگ
    static std::string format_large_number(uint64_t number);
    static std::string format_hashrate(uint64_t hashrate);
    static std::string format_percentage(double percentage);
    static std::string format_duration(uint64_t seconds);
    
    // تبدیل هگزادسیمال
    static std::string bytes_to_hex(const std::vector<uint8_t>& bytes);
    static std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    static std::string bytes_to_hex(const uint8_t* data, size_t length);
    
    // فرمت آدرس‌ها
    static std::string format_address(const std::string& address);
    static std::string shorten_address(const std::string& address, size_t keep_chars = 8);
    
    // فرمت زمان
    static std::string get_current_timestamp();
    static std::string format_timestamp(uint64_t timestamp);
    static std::string format_duration(std::chrono::seconds duration);
    
    // رنگ‌های ترمینال (برای گزارش‌گیری)
    static std::string color_red(const std::string& text);
    static std::string color_green(const std::string& text);
    static std::string color_yellow(const std::string& text);
    static std::string color_blue(const std::string& text);
    static std::string color_magenta(const std::string& text);
    static std::string color_cyan(const std::string& text);
    static std::string color_white(const std::string& text);
    static std::string color_bold(const std::string& text);
    static std::string color_reset();
    
    // فرمت JSON
    static std::string to_json_string(const std::string& value);
    static std::string escape_json(const std::string& input);
    
    // فرمت CSV
    static std::string escape_csv(const std::string& input);
    
    // کاربردی
    static std::string format_size_bytes(size_t bytes);
    static std::string format_time_remaining(uint64_t processed, uint64_t total, uint64_t speed);
    static std::string format_progress_bar(double percentage, int width = 50);

private:
    FormatUtils() = delete;
    ~FormatUtils() = delete;
    
    // غیرقابل کپی
    FormatUtils(const FormatUtils&) = delete;
    FormatUtils& operator=(const FormatUtils&) = delete;
};

} // namespace bitcoin_miner

#endif // FORMAT_UTILS_H