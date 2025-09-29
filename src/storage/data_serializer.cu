#include "data_serializer.h"
#include "../utils/logger.h"
#include "../utils/format_utils.h"
#include <sstream>
#include <iomanip>
#include <zlib.h>

namespace bitcoin_miner {

class DataSerializer::Impl {
public:
    Impl(Format format) : format(format), compression_enabled(false) {}
    
    std::string serialize(const MiningResult& result) const {
        switch (format) {
            case Format::JSON:
                return serialize_to_json(result);
            case Format::CSV:
                return serialize_to_csv(result);
            case Format::BINARY:
                return serialize_to_binary(result);
            case Format::XML:
                return serialize_to_xml(result);
            default:
                return serialize_to_json(result);
        }
    }
    
    std::string serialize_batch(const std::vector<MiningResult>& results) const {
        switch (format) {
            case Format::JSON:
                return serialize_batch_to_json(results);
            case Format::CSV:
                return serialize_batch_to_csv(results);
            case Format::BINARY:
                return serialize_batch_to_binary(results);
            case Format::XML:
                return serialize_batch_to_xml(results);
            default:
                return serialize_batch_to_json(results);
        }
    }
    
    MiningResult deserialize(const std::string& data) const {
        switch (format) {
            case Format::JSON:
                return deserialize_from_json(data);
            case Format::CSV:
                return deserialize_from_csv(data);
            case Format::BINARY:
                return deserialize_from_binary(data);
            case Format::XML:
                return deserialize_from_xml(data);
            default:
                return deserialize_from_json(data);
        }
    }
    
    std::vector<MiningResult> deserialize_batch(const std::string& data) const {
        switch (format) {
            case Format::JSON:
                return deserialize_batch_from_json(data);
            case Format::CSV:
                return deserialize_batch_from_csv(data);
            case Format::BINARY:
                return deserialize_batch_from_binary(data);
            case Format::XML:
                return deserialize_batch_from_xml(data);
            default:
                return deserialize_batch_from_json(data);
        }
    }
    
    bool save_to_file(const std::string& filename, const std::vector<MiningResult>& results) const {
        try {
            std::string data = serialize_batch(results);
            
            if (compression_enabled) {
                auto compressed_data = compress(data);
                std::string compressed_filename = filename + ".gz";
                std::ofstream file(compressed_filename, std::ios::binary);
                if (!file.is_open()) {
                    Logger::error("Cannot create compressed file: {}", compressed_filename);
                    return false;
                }
                file.write(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
                file.close();
                Logger::info("Saved {} compressed results to {}", results.size(), compressed_filename);
            } else {
                std::ofstream file(filename);
                if (!file.is_open()) {
                    Logger::error("Cannot create file: {}", filename);
                    return false;
                }
                file << data;
                file.close();
                Logger::info("Saved {} results to {}", results.size(), filename);
            }
            
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to save to file: {}", e.what());
            return false;
        }
    }
    
    std::vector<MiningResult> load_from_file(const std::string& filename) const {
        try {
            std::string data;
            std::string actual_filename = filename;
            
            // Check if file is compressed
            if (filename.size() > 3 && filename.substr(filename.size() - 3) == ".gz") {
                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    Logger::error("Cannot open compressed file: {}", filename);
                    return {};
                }
                
                std::vector<uint8_t> compressed_data(
                    (std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>()
                );
                
                data = decompress(compressed_data);
                actual_filename = filename.substr(0, filename.size() - 3);
            } else {
                std::ifstream file(filename);
                if (!file.is_open()) {
                    Logger::error("Cannot open file: {}", filename);
                    return {};
                }
                
                data.assign(
                    (std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>()
                );
            }
            
            // Determine format from filename extension if not specified
            Format file_format = format;
            if (actual_filename.size() > 4) {
                std::string ext = actual_filename.substr(actual_filename.size() - 4);
                if (ext == ".csv") file_format = Format::CSV;
                else if (ext == ".xml") file_format = Format::XML;
                else if (ext == ".bin") file_format = Format::BINARY;
                // else keep current format (usually JSON)
            }
            
            std::vector<MiningResult> results;
            if (file_format == Format::JSON) {
                results = deserialize_batch_from_json(data);
            } else if (file_format == Format::CSV) {
                results = deserialize_batch_from_csv(data);
            } else if (file_format == Format::BINARY) {
                results = deserialize_batch_from_binary(data);
            } else if (file_format == Format::XML) {
                results = deserialize_batch_from_xml(data);
            }
            
            Logger::info("Loaded {} results from {}", results.size(), filename);
            return results;
            
        } catch (const std::exception& e) {
            Logger::error("Failed to load from file: {}", e.what());
            return {};
        }
    }
    
    void set_format(Format new_format) {
        format = new_format;
        const char* format_names[] = {"JSON", "CSV", "BINARY", "XML"};
        Logger::info("Data format set to {}", format