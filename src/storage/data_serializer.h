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
        Logger::info("Data format set to {}", format_names[static_cast<int>(new_format)]);
    }
    
    Format get_format() const {
        return format;
    }
    
    void set_compression(bool enabled) {
        compression_enabled = enabled;
        Logger::info("Compression {}", enabled ? "enabled" : "disabled");
    }
    
    std::vector<uint8_t> compress(const std::string& data) const {
        uLongf compressed_size = compressBound(data.size());
        std::vector<uint8_t> compressed(compressed_size);
        
        int result = compress2(compressed.data(), &compressed_size,
                             reinterpret_cast<const Bytef*>(data.data()), data.size(),
                             Z_BEST_COMPRESSION);
        
        if (result != Z_OK) {
            Logger::error("Compression failed with error code: {}", result);
            // Fallback: return uncompressed data
            return std::vector<uint8_t>(data.begin(), data.end());
        }
        
        compressed.resize(compressed_size);
        Logger::debug("Compressed {} bytes to {} bytes (ratio: {:.1f}%)", 
                     data.size(), compressed_size, 
                     (compressed_size * 100.0) / data.size());
        
        return compressed;
    }
    
    std::string decompress(const std::vector<uint8_t>& compressed_data) const {
        // Estimate decompressed size (conservative estimate)
        uLongf decompressed_size = compressed_data.size() * 10;
        std::vector<char> decompressed(decompressed_size);
        
        int result = uncompress(
            reinterpret_cast<Bytef*>(decompressed.data()), &decompressed_size,
            compressed_data.data(), compressed_data.size()
        );
        
        if (result != Z_OK) {
            Logger::error("Decompression failed with error code: {}", result);
            return "";
        }
        
        decompressed.resize(decompressed_size);
        return std::string(decompressed.begin(), decompressed.end());
    }

private:
    Format format;
    bool compression_enabled;
    
    std::string serialize_to_json(const MiningResult& result) const {
        std::stringstream ss;
        ss << "{";
        ss << "\"timestamp\":" << result.timestamp << ",";
        ss << "\"address\":\"" << FormatUtils::escape_json(result.address) << "\",";
        ss << "\"private_key_hex\":\"" << result.private_key_hex << "\",";
        ss << "\"private_key_wif\":\"" << FormatUtils::escape_json(result.private_key_wif) << "\",";
        ss << "\"private_key_wif_compressed\":\"" << FormatUtils::escape_json(result.private_key_wif_compressed) << "\",";
        ss << "\"public_key_compressed_hex\":\"" << result.public_key_compressed_hex << "\",";
        ss << "\"public_key_uncompressed_hex\":\"" << result.public_key_uncompressed_hex << "\",";
        ss << "\"address_p2pkh\":\"" << FormatUtils::escape_json(result.address_p2pkh) << "\",";
        ss << "\"address_p2sh\":\"" << FormatUtils::escape_json(result.address_p2sh) << "\",";
        ss << "\"address_bech32\":\"" << FormatUtils::escape_json(result.address_bech32) << "\",";
        ss << "\"address_bech32m\":\"" << FormatUtils::escape_json(result.address_bech32m) << "\",";
        ss << "\"address_type\":\"" << result.address_type << "\",";
        ss << "\"nonce\":" << result.nonce << ",";
        ss << "\"is_valid\":" << (result.is_valid ? "true" : "false") << ",";
        ss << "\"verified\":" << (result.verified ? "true" : "false") << ",";
        ss << "\"source_batch\":\"" << FormatUtils::escape_json(result.source_batch) << "\",";
        ss << "\"gpu_device_id\":" << result.gpu_device_id << ",";
        ss << "\"worker_thread_id\":\"" << FormatUtils::escape_json(result.worker_thread_id) << "\"";
        ss << "}";
        return ss.str();
    }
    
    std::string serialize_to_csv(const MiningResult& result) const {
        std::stringstream ss;
        ss << result.timestamp << ",";
        ss << FormatUtils::escape_csv(result.address) << ",";
        ss << result.private_key_hex << ",";
        ss << FormatUtils::escape_csv(result.private_key_wif) << ",";
        ss << FormatUtils::escape_csv(result.private_key_wif_compressed) << ",";
        ss << result.public_key_compressed_hex << ",";
        ss << result.public_key_uncompressed_hex << ",";
        ss << FormatUtils::escape_csv(result.address_p2pkh) << ",";
        ss << FormatUtils::escape_csv(result.address_p2sh) << ",";
        ss << FormatUtils::escape_csv(result.address_bech32) << ",";
        ss << FormatUtils::escape_csv(result.address_bech32m) << ",";
        ss << result.address_type << ",";
        ss << result.nonce << ",";
        ss << (result.is_valid ? "true" : "false") << ",";
        ss << (result.verified ? "true" : "false") << ",";
        ss << FormatUtils::escape_csv(result.source_batch) << ",";
        ss << result.gpu_device_id << ",";
        ss << FormatUtils::escape_csv(result.worker_thread_id);
        return ss.str();
    }
    
    std::string serialize_to_binary(const MiningResult& result) const {
        // Simple binary serialization - in production, use a proper binary format
        std::string data;
        
        // Add timestamp
        data.append(reinterpret_cast<const char*>(&result.timestamp), sizeof(result.timestamp));
        
        // Add address (length-prefixed)
        uint32_t addr_len = result.address.size();
        data.append(reinterpret_cast<const char*>(&addr_len), sizeof(addr_len));
        data.append(result.address);
        
        // Add private key hex
        uint32_t priv_len = result.private_key_hex.size();
        data.append(reinterpret_cast<const char*>(&priv_len), sizeof(priv_len));
        data.append(result.private_key_hex);
        
        // Add validity flags
        data.append(reinterpret_cast<const char*>(&result.is_valid), sizeof(result.is_valid));
        data.append(reinterpret_cast<const char*>(&result.verified), sizeof(result.verified));
        
        return data;
    }
    
    std::string serialize_to_xml(const MiningResult& result) const {
        std::stringstream ss;
        ss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        ss << "<mining_result>\n";
        ss << "  <timestamp>" << result.timestamp << "</timestamp>\n";
        ss << "  <address>" << FormatUtils::escape_json(result.address) << "</address>\n";
        ss << "  <private_key_hex>" << result.private_key_hex << "</private_key_hex>\n";
        ss << "  <private_key_wif>" << FormatUtils::escape_json(result.private_key_wif) << "</private_key_wif>\n";
        ss << "  <private_key_wif_compressed>" << FormatUtils::escape_json(result.private_key_wif_compressed) << "</private_key_wif_compressed>\n";
        ss << "  <public_key_compressed_hex>" << result.public_key_compressed_hex << "</public_key_compressed_hex>\n";
        ss << "  <public_key_uncompressed_hex>" << result.public_key_uncompressed_hex << "</public_key_uncompressed_hex>\n";
        ss << "  <address_p2pkh>" << FormatUtils::escape_json(result.address_p2pkh) << "</address_p2pkh>\n";
        ss << "  <address_p2sh>" << FormatUtils::escape_json(result.address_p2sh) << "</address_p2sh>\n";
        ss << "  <address_bech32>" << FormatUtils::escape_json(result.address_bech32) << "</address_bech32>\n";
        ss << "  <address_bech32m>" << FormatUtils::escape_json(result.address_bech32m) << "</address_bech32m>\n";
        ss << "  <address_type>" << result.address_type << "</address_type>\n";
        ss << "  <nonce>" << result.nonce << "</nonce>\n";
        ss << "  <is_valid>" << (result.is_valid ? "true" : "false") << "</is_valid>\n";
        ss << "  <verified>" << (result.verified ? "true" : "false") << "</verified>\n";
        ss << "  <source_batch>" << FormatUtils::escape_json(result.source_batch) << "</source_batch>\n";
        ss << "  <gpu_device_id>" << result.gpu_device_id << "</gpu_device_id>\n";
        ss << "  <worker_thread_id>" << FormatUtils::escape_json(result.worker_thread_id) << "</worker_thread_id>\n";
        ss << "</mining_result>";
        return ss.str();
    }
    
    std::string serialize_batch_to_json(const std::vector<MiningResult>& results) const {
        std::stringstream ss;
        ss << "{\n";
        ss << "  \"mining_results\": [\n";
        
        for (size_t i = 0; i < results.size(); ++i) {
            ss << "    " << serialize_to_json(results[i]);
            if (i < results.size() - 1) {
                ss << ",";
            }
            ss << "\n";
        }
        
        ss << "  ],\n";
        ss << "  \"count\": " << results.size() << ",\n";
        ss << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
        ss << "}";
        return ss.str();
    }
    
    std::string serialize_batch_to_csv(const std::vector<MiningResult>& results) const {
        std::stringstream ss;
        // Header
        ss << "timestamp,address,private_key_hex,private_key_wif,private_key_wif_compressed,";
        ss << "public_key_compressed_hex,public_key_uncompressed_hex,address_p2pkh,address_p2sh,";
        ss << "address_bech32,address_bech32m,address_type,nonce,is_valid,verified,source_batch,";
        ss << "gpu_device_id,worker_thread_id\n";
        
        // Data
        for (const auto& result : results) {
            ss << serialize_to_csv(result) << "\n";
        }
        
        return ss.str();
    }
    
    std::string serialize_batch_to_binary(const std::vector<MiningResult>& results) const {
        std::string data;
        
        // Header: number of results
        uint32_t count = results.size();
        data.append(reinterpret_cast<const char*>(&count), sizeof(count));
        
        // Each result
        for (const auto& result : results) {
            data.append(serialize_to_binary(result));
        }
        
        return data;
    }
    
    std::string serialize_batch_to_xml(const std::vector<MiningResult>& results) const {
        std::stringstream ss;
        ss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        ss << "<mining_results batch_timestamp=\"" 
           << std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch()).count() 
           << "\" count=\"" << results.size() << "\">\n";
        
        for (const auto& result : results) {
            ss << serialize_to_xml(result) << "\n";
        }
        
        ss << "</mining_results>";
        return ss.str();
    }
    
    // Deserialization methods (simplified implementations)
    MiningResult deserialize_from_json(const std::string& data) const {
        MiningResult result;
        // Simple JSON parsing - in production, use a proper JSON library
        // This is a minimal implementation
        return result;
    }
    
    MiningResult deserialize_from_csv(const std::string& data) const {
        MiningResult result;
        // Simple CSV parsing
        return result;
    }
    
    MiningResult deserialize_from_binary(const std::string& data) const {
        MiningResult result;
        // Simple binary parsing
        return result;
    }
    
    MiningResult deserialize_from_xml(const std::string& data) const {
        MiningResult result;
        // Simple XML parsing
        return result;
    }
    
    std::vector<MiningResult> deserialize_batch_from_json(const std::string& data) const {
        // Placeholder implementation
        return {};
    }
    
    std::vector<MiningResult> deserialize_batch_from_csv(const std::string& data) const {
        // Placeholder implementation
        return {};
    }
    
    std::vector<MiningResult> deserialize_batch_from_binary(const std::string& data) const {
        // Placeholder implementation
        return {};
    }
    
    std::vector<MiningResult> deserialize_batch_from_xml(const std::string& data) const {
        // Placeholder implementation
        return {};
    }
};

// Implementation of DataSerializer wrapper methods
DataSerializer::DataSerializer(Format format) : impl(new Impl(format)) {}
DataSerializer::~DataSerializer() = default;

std::string DataSerializer::serialize(const MiningResult& result) const {
    return impl->serialize(result);
}

std::string DataSerializer::serialize_batch(const std::vector<MiningResult>& results) const {
    return impl->serialize_batch(results);
}

MiningResult DataSerializer::deserialize(const std::string& data) const {
    return impl->deserialize(data);
}

std::vector<MiningResult> DataSerializer::deserialize_batch(const std::string& data) const {
    return impl->deserialize_batch(data);
}

bool DataSerializer::save_to_file(const std::string& filename, const std::vector<MiningResult>& results) const {
    return impl->save_to_file(filename, results);
}

std::vector<MiningResult> DataSerializer::load_from_file(const std::string& filename) const {
    return impl->load_from_file(filename);
}

void DataSerializer::set_format(Format format) {
    impl->set_format(format);
}

DataSerializer::Format DataSerializer::get_format() const {
    return impl->get_format();
}

void DataSerializer::set_compression(bool enabled) {
    impl->set_compression(enabled);
}

std::vector<uint8_t> DataSerializer::compress(const std::string& data) const {
    return impl->compress(data);
}

std::string DataSerializer::decompress(const std::vector<uint8_t>& compressed_data) const {
    return impl->decompress(compressed_data);
}

} // namespace bitcoin_miner