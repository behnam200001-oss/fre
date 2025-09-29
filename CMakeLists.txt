cmake_minimum_required(VERSION 3.18)
project(AdvancedBitcoinMiner CUDA CXX)

# تنظیمات پایه
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# تنظیمات بهینه‌سازی
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -O3,-v --use_fast_math -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

# یافتن بستگی‌ها
find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# مدیریت کتابخانه‌های سوم
include(FetchContent)

# secp256k1 library
FetchContent_Declare(
    secp256k1
    GIT_REPOSITORY https://github.com/bitcoin-core/secp256k1.git
    GIT_TAG master
)

# Google Test برای تست‌ها
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(secp256k1 googletest)

# کامپایل secp256k1 به عنوان کتابخانه استاتیک
add_library(secp256k1_lib STATIC
    third_party/secp256k1/src/secp256k1.c
)

target_include_directories(secp256k1_lib 
    PRIVATE 
    third_party/secp256k1
    third_party/secp256k1/include
    third_party/secp256k1/src
)

target_compile_definitions(secp256k1_lib 
    PRIVATE 
    -DUSE_NUM_NONE=1
    -DUSE_FIELD_INV_BUILTIN=1
    -DUSE_SCALAR_INV_BUILTIN=1
    -DHAVE_BUILTIN_EXPECT=1
    -DUSE_ECMULT_STATIC_PRECOMPUTATION=1
    -DUSE_ENDOMORPHISM=1
    -DENABLE_MODULE_RECOVERY=1
)

# فایل‌های منبع
file(GLOB_RECURSE SRC_FILES 
    "src/*.cu" 
    "src/*.cpp" 
    "src/*.c"
    "kernels/*.cu"
)

# کتابخانه اصلی
add_library(bitcoin_miner_cuda STATIC ${SRC_FILES})

# هدف اجرایی
add_executable(advanced_bitcoin_miner 
    src/main.cu
    $<TARGET_OBJECTS:bitcoin_miner_cuda>
)

# لینک کتابخانه‌ها
target_link_libraries(advanced_bitcoin_miner 
    PRIVATE 
    bitcoin_miner_cuda
    secp256k1_lib
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    OpenSSL::Crypto
    OpenSSL::SSL
    Threads::Threads
    gtest
    gtest_main
    dl
    pthread
    rt
)

# شامل‌ها
target_include_directories(advanced_bitcoin_miner 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${OPENSSL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/secp256k1/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/secp256k1/src
)

# تنظیمات خاص CUDA
set_target_properties(advanced_bitcoin_miner PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "70;75;80;86"
)

# تست‌ها
enable_testing()
add_subdirectory(tests)

# نصب
install(TARGETS advanced_bitcoin_miner DESTINATION bin)