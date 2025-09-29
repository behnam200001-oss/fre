#include "device_properties.h"
#include "../utils/logger.h"

namespace bitcoin_miner {

int DeviceProperties::get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        Logger::error("Failed to get device count: {}", cudaGetErrorString(err));
        return 0;
    }
    return count;
}

DeviceProperties::DeviceInfo DeviceProperties::get_device_properties(int device_id) {
    DeviceInfo info;
    cudaDeviceProp props;
    
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        Logger::error("Failed to get device properties for device {}: {}", 
                     device_id, cudaGetErrorString(err));
        return info;
    }
    
    info.name = props.name;
    info.totalGlobalMem = props.totalGlobalMem;
    info.multiProcessorCount = props.multiProcessorCount;
    info.maxThreadsPerBlock = props.maxThreadsPerBlock;
    info.major = props.major;
    info.minor = props.minor;
    
    return info;
}

} // namespace bitcoin_miner