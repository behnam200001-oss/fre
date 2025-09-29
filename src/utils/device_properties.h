#pragma once
#ifndef DEVICE_PROPERTIES_H
#define DEVICE_PROPERTIES_H

#include <cuda_runtime.h>
#include <string>

namespace bitcoin_miner {

class DeviceProperties {
public:
    struct DeviceInfo {
        std::string name;
        size_t totalGlobalMem;
        int multiProcessorCount;
        int maxThreadsPerBlock;
        int major;
        int minor;
    };
    
    static int get_device_count();
    static DeviceInfo get_device_properties(int device_id);
};

} // namespace bitcoin_miner

#endif // DEVICE_PROPERTIES_H