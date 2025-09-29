#include "statistics_manager.h"
#include "../utils/logger.h"
#include "../utils/format_utils.h"
#include <fstream>
#include <numeric>
#include <algorithm>

namespace bitcoin_miner {

class StatisticsManager::Impl {
public:
    Impl() : retention_period(std::chrono::hours(24)) {
        // Initialize with current time
        start_time = std::chrono::system_clock::now();
    }
    
    void record_gpu_metric(int device_id, const std::string& metric, double value) {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        auto key = "gpu_" + std::to_string(device_id) + "_" + metric;
        record_metric(key, value);
        
        Logger::debug("Recorded GPU metric: {} = {:.2f}", key, value);
    }
    
    void record_system_metric(const std::string& metric, double value) {
        std::lock_guard<std::mutex> lock(data_mutex);
        record_metric("system_" + metric, value);
        Logger::debug("Recorded system metric: {} = {:.2f}", metric, value);
    }
    
    void record_match_found(const MiningResult& result) {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        TimeSeriesPoint point;
        point.timestamp = std::chrono::system_clock::now();
        point.value = 1.0; // Count as 1 match
        
        matches_found.push_back(point);
        
        // Also record as a metric for time series analysis
        record_metric("matches_found", 1.0);
        
        // Keep only recent data
        if (matches_found.size() > MAX_MATCH_HISTORY) {
            matches_found.pop_front();
        }
        
        Logger::info("Recorded match: {}", result.address);
    }
    
    std::vector<TimeSeriesPoint> get_metric_history(const std::string& metric, 
                                                   std::chrono::seconds duration) const {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        auto it = time_series_data.find(metric);
        if (it == time_series_data.end()) {
            return {};
        }
        
        auto cutoff = std::chrono::system_clock::now() - duration;
        std::vector<TimeSeriesPoint> result;
        
        for (const auto& point : it->second) {
            if (point.timestamp >= cutoff) {
                result.push_back(point);
            }
        }
        
        return result;
    }
    
    GPUStats get_gpu_stats(int device_id, std::chrono::seconds duration) const {
        GPUStats stats;
        stats.device_id = device_id;
        
        stats.utilization = get_metric_history("gpu_" + std::to_string(device_id) + "_utilization", duration);
        stats.temperature = get_metric_history("gpu_" + std::to_string(device_id) + "_temperature", duration);
        stats.memory_usage = get_metric_history("gpu_" + std::to_string(device_id) + "_memory_usage", duration);
        stats.hashrate = get_metric_history("gpu_" + std::to_string(device_id) + "_hashrate", duration);
        
        return stats;
    }
    
    SystemStats get_system_stats(std::chrono::seconds duration) const {
        SystemStats stats;
        
        stats.total_hashrate = get_metric_history("system_total_hashrate", duration);
        stats.matches_found = get_metric_history("system_matches_found", duration);
        stats.false_positives = get_metric_history("system_false_positives", duration);
        stats.efficiency = get_metric_history("system_efficiency", duration);
        
        return stats;
    }
    
    double calculate_average(const std::string& metric, std::chrono::seconds duration) const {
        auto history = get_metric_history(metric, duration);
        if (history.empty()) return 0.0;
        
        double sum = 0.0;
        for (const auto& point : history) {
            sum += point.value;
        }
        
        return sum / history.size();
    }
    
    double calculate_peak(const std::string& metric, std::chrono::seconds duration) const {
        auto history = get_metric_history(metric, duration);
        if (history.empty()) return 0.0;
        
        double peak = history[0].value;
        for (const auto& point : history) {
            if (point.value > peak) {
                peak = point.value;
            }
        }
        
        return peak;
    }
    
    std::map<std::string, double> get_performance_metrics() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        std::map<std::string, double> metrics;
        auto now = std::chrono::system_clock::now();
        auto cutoff = now - std::chrono::hours(1); // Last hour
        
        for (const auto& pair : time_series_data) {
            double sum = 0.0;
            int count = 0;
            
            for (const auto& point : pair.second) {
                if (point.timestamp >= cutoff) {
                    sum += point.value;
                    count++;
                }
            }
            
            if (count > 0) {
                metrics[pair.first] = sum / count;
            }
        }
        
        // Add match statistics
        auto match_history = get_matches_last_hour();
        metrics["matches_per_hour"] = match_history.size();
        
        return metrics;
    }
    
    std::string generate_summary_report() const {
        std::stringstream ss;
        
        auto metrics = get_performance_metrics();
        auto system_stats = get_system_stats(std::chrono::hours(1));
        
        ss << "Advanced Bitcoin Miner - Performance Summary Report\n";
        ss << "===================================================\n";
        ss << "Generated: " << FormatUtils::get_current_timestamp() << "\n";
        ss << "Uptime: " << FormatUtils::format_duration(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - start_time)) << "\n\n";
        
        ss << "SYSTEM METRICS (1-hour average):\n";
        ss << "  Total Hashrate: " << FormatUtils::format_hashrate(
            metrics.count("system_total_hashrate") ? metrics.at("system_total_hashrate") : 0) << "\n";
        ss << "  Efficiency: " << std::fixed << std::setprecision(2) 
           << (metrics.count("system_efficiency") ? metrics.at("system_efficiency") * 100 : 0) << "%\n";
        ss << "  Matches Found: " << (metrics.count("matches_per_hour") ? metrics.at("matches_per_hour") : 0) << "\n\n";
        
        ss << "GPU METRICS (1-hour average):\n";
        for (const auto& metric : metrics) {
            if (metric.first.find("gpu_") == 0) {
                size_t pos = metric.first.find("_", 4);
                if (pos != std::string::npos) {
                    std::string gpu_id = metric.first.substr(4, pos - 4);
                    std::string metric_name = metric.first.substr(pos + 1);
                    
                    ss << "  GPU " << gpu_id << " " << metric_name << ": " << std::fixed << std::setprecision(2);
                    
                    if (metric_name == "utilization" || metric_name == "efficiency") {
                        ss << (metric.second * 100) << "%";
                    } else if (metric_name == "hashrate") {
                        ss << FormatUtils::format_hashrate(metric.second);
                    } else if (metric_name == "temperature") {
                        ss << metric.second << "°C";
                    } else {
                        ss << metric.second;
                    }
                    
                    ss << "\n";
                }
            }
        }
        
        if (!system_stats.matches_found.empty()) {
            double matches_per_hour = calculate_average("system_matches_found", std::chrono::hours(1)) * 3600;
            ss << "\nPRODUCTION STATISTICS:\n";
            ss << "  Matches found per hour: " << std::fixed << std::setprecision(2) << matches_per_hour << "\n";
            
            if (!matches_found.empty()) {
                auto first_match = matches_found.front();
                auto last_match = matches_found.back();
                auto total_duration = std::chrono::duration_cast<std::chrono::hours>(
                    last_match.timestamp - first_match.timestamp).count();
                
                if (total_duration > 0) {
                    double avg_matches_per_hour = static_cast<double>(matches_found.size()) / total_duration;
                    ss << "  Average matches per hour: " << std::fixed << std::setprecision(2) << avg_matches_per_hour << "\n";
                }
            }
        }
        
        ss << "\nPERFORMANCE INSIGHTS:\n";
        // Add performance insights based on collected data
        double avg_efficiency = metrics.count("system_efficiency") ? metrics.at("system_efficiency") : 0;
        if (avg_efficiency > 0.8) {
            ss << "  ✅ System is operating efficiently\n";
        } else if (avg_efficiency > 0.5) {
            ss << "  ⚠️  System efficiency could be improved\n";
        } else {
            ss << "  ❌ System efficiency is low - consider optimization\n";
        }
        
        return ss.str();
    }
    
    bool export_to_csv(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        try {
            std::ofstream file(filename);
            if (!file.is_open()) {
                Logger::error("Cannot create CSV file: {}", filename);
                return false;
            }
            
            file << "timestamp,metric,value\n";
            
            for (const auto& pair : time_series_data) {
                for (const auto& point : pair.second) {
                    file << std::chrono::duration_cast<std::chrono::seconds>(
                        point.timestamp.time_since_epoch()).count() << ",";
                    file << pair.first << ",";
                    file << point.value << "\n";
                }
            }
            
            // Also export match history
            for (const auto& point : matches_found) {
                file << std::chrono::duration_cast<std::chrono::seconds>(
                    point.timestamp.time_since_epoch()).count() << ",";
                file << "match_found" << ",";
                file << point.value << "\n";
            }
            
            file.close();
            Logger::info("Exported {} data points to CSV: {}", 
                        time_series_data.size() + matches_found.size(), filename);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to export to CSV: {}", e.what());
            return false;
        }
    }
    
    bool export_to_json(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        try {
            std::ofstream file(filename);
            if (!file.is_open()) {
                Logger::error("Cannot create JSON file: {}", filename);
                return false;
            }
            
            file << "{\n";
            file << "  \"statistics\": {\n";
            
            bool first_metric = true;
            for (const auto& pair : time_series_data) {
                if (!first_metric) {
                    file << ",\n";
                }
                first_metric = false;
                
                file << "    \"" << pair.first << "\": [\n";
                
                bool first_point = true;
                for (const auto& point : pair.second) {
                    if (!first_point) {
                        file << ",\n";
                    }
                    first_point = false;
                    
                    file << "      {\n";
                    file << "        \"timestamp\": " 
                         << std::chrono::duration_cast<std::chrono::seconds>(
                             point.timestamp.time_since_epoch()).count() << ",\n";
                    file << "        \"value\": " << point.value << "\n";
                    file << "      }";
                }
                
                file << "\n    ]";
            }
            
            file << "\n  },\n";
            file << "  \"matches\": [\n";
            
            bool first_match = true;
            for (const auto& point : matches_found) {
                if (!first_match) {
                    file << ",\n";
                }
                first_match = false;
                
                file << "    {\n";
                file << "      \"timestamp\": " 
                     << std::chrono::duration_cast<std::chrono::seconds>(
                         point.timestamp.time_since_epoch()).count() << "\n";
                file << "    }";
            }
            
            file << "\n  ],\n";
            file << "  \"metadata\": {\n";
            file << "    \"total_data_points\": " << time_series_data.size() << ",\n";
            file << "    \"total_matches\": " << matches_found.size() << ",\n";
            file << "    \"export_timestamp\": " 
                 << std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
            file << "  }\n";
            file << "}\n";
            
            file.close();
            Logger::info("Exported statistics to JSON: {}", filename);
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to export to JSON: {}", e.what());
            return false;
        }
    }
    
    void set_retention_period(std::chrono::hours period) {
        retention_period = period;
        Logger::info("Retention period set to {} hours", period.count());
    }
    
    void cleanup_old_data() {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        auto cutoff = std::chrono::system_clock::now() - retention_period;
        size_t total_removed = 0;
        
        for (auto& pair : time_series_data) {
            auto& points = pair.second;
            auto it = std::remove_if(points.begin(), points.end(),
                [cutoff](const TimeSeriesPoint& point) {
                    return point.timestamp < cutoff;
                });
            
            total_removed += std::distance(it, points.end());
            points.erase(it, points.end());
        }
        
        // Cleanup matches_found
        auto it = std::remove_if(matches_found.begin(), matches_found.end(),
            [cutoff](const TimeSeriesPoint& point) {
                return point.timestamp < cutoff;
            });
        total_removed += std::distance(it, matches_found.end());
        matches_found.erase(it, matches_found.end());
        
        if (total_removed > 0) {
            Logger::info("Cleaned up {} old data points", total_removed);
        }
    }

private:
    static constexpr size_t MAX_MATCH_HISTORY = 10000;
    static constexpr size_t MAX_METRIC_POINTS = 50000;
    
    mutable std::mutex data_mutex;
    std::map<std::string, std::vector<TimeSeriesPoint>> time_series_data;
    std::deque<TimeSeriesPoint> matches_found;
    std::chrono::hours retention_period;
    std::chrono::system_clock::time_point start_time;
    
    void record_metric(const std::string& key, double value) {
        TimeSeriesPoint point;
        point.timestamp = std::chrono::system_clock::now();
        point.value = value;
        
        time_series_data[key].push_back(point);
        
        // Limit the size of time series data
        if (time_series_data[key].size() > MAX_METRIC_POINTS) {
            time_series_data[key].erase(time_series_data[key].begin());
        }
    }
    
    std::vector<TimeSeriesPoint> get_matches_last_hour() const {
        auto cutoff = std::chrono::system_clock::now() - std::chrono::hours(1);
        std::vector<TimeSeriesPoint> result;
        
        for (const auto& point : matches_found) {
            if (point.timestamp >= cutoff) {
                result.push_back(point);
            }
        }
        
        return result;
    }
};

// Implementation of StatisticsManager wrapper methods
StatisticsManager::StatisticsManager() : impl(new Impl()) {}
StatisticsManager::~StatisticsManager() = default;

void StatisticsManager::record_gpu_metric(int device_id, const std::string& metric, double value) {
    impl->record_gpu_metric(device_id, metric, value);
}

void StatisticsManager::record_system_metric(const std::string& metric, double value) {
    impl->record_system_metric(metric, value);
}

void StatisticsManager::record_match_found(const MiningResult& result) {
    impl->record_match_found(result);
}

std::vector<StatisticsManager::TimeSeriesPoint> StatisticsManager::get_metric_history(
    const std::string& metric, std::chrono::seconds duration) const {
    return impl->get_metric_history(metric, duration);
}

StatisticsManager::GPUStats StatisticsManager::get_gpu_stats(int device_id, std::chrono::seconds duration) const {
    return impl->get_gpu_stats(device_id, duration);
}

StatisticsManager::SystemStats StatisticsManager::get_system_stats(std::chrono::seconds duration) const {
    return impl->get_system_stats(duration);
}

double StatisticsManager::calculate_average(const std::string& metric, std::chrono::seconds duration) const {
    return impl->calculate_average(metric, duration);
}

double StatisticsManager::calculate_peak(const std::string& metric, std::chrono::seconds duration) const {
    return impl->calculate_peak(metric, duration);
}

std::map<std::string, double> StatisticsManager::get_performance_metrics() const {
    return impl->get_performance_metrics();
}

std::string StatisticsManager::generate_summary_report() const {
    return impl->generate_summary_report();
}

bool StatisticsManager::export_to_csv(const std::string& filename) const {
    return impl->export_to_csv(filename);
}

bool StatisticsManager::export_to_json(const std::string& filename) const {
    return impl->export_to_json(filename);
}

void StatisticsManager::set_retention_period(std::chrono::hours period) {
    impl->set_retention_period(period);
}

void StatisticsManager::cleanup_old_data() {
    impl->cleanup_old_data();
}

} // namespace bitcoin_miner