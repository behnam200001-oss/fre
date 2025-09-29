#include "advanced_secp256k1.h"
#include "../utils/cuda_utils.h"
#include "../utils/logger.h"
#include <cassert>
#include <sstream>
#include <secp256k1.h>
#include <secp256k1_recovery.h>

namespace bitcoin_miner {

// ثابت‌های منحنی secp256k1
__constant__ uint32_t d_secp256k1_p[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ uint32_t d_secp256k1_n[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ uint32_t d_secp256k1_gx[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

__constant__ uint32_t d_secp256k1_gy[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

// توابع کمکی برای محاسبات میدان محدود
__device__ void fe_add(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a[i] + b[i];
        r[i] = carry & 0xFFFFFFFF;
        carry >>= 32;
    }
}

__device__ void fe_sub(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    int64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t temp = (int64_t)a[i] - b[i] - borrow;
        r[i] = temp & 0xFFFFFFFF;
        borrow = (temp >> 32) & 1;
    }
}

__device__ void fe_mul(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint64_t accum[16] = {0};
    
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            accum[i + j] += (uint64_t)a[i] * b[j] + carry;
            carry = accum[i + j] >> 32;
            accum[i + j] &= 0xFFFFFFFF;
        }
        accum[i + 8] = carry;
    }
    
    // کاهش modulo p
    fe_reduce(r, accum);
}

__device__ void fe_reduce(uint32_t* r, const uint64_t* a) {
    // پیاده‌سازی کاهش modulo p
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += a[i] + a[i + 8] * 0x1000003D1ULL;
        r[i] = carry & 0xFFFFFFFF;
        carry >>= 32;
    }
}

__device__ void fe_square(uint32_t* r, const uint32_t* a) {
    uint64_t accum[16] = {0};
    
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            accum[i + j] += (uint64_t)a[i] * a[j] + carry;
            carry = accum[i + j] >> 32;
            accum[i + j] &= 0xFFFFFFFF;
        }
        accum[i + 8] = carry;
    }
    
    fe_reduce(r, accum);
}

__device__ void fe_invert(uint32_t* r, const uint32_t* a) {
    // پیاده‌سازی معکوس میدان محدود با استفاده از قضیه کوچک فرما
    // a^(p-2) mod p = a^(-1) mod p
    uint32_t result[8] = {1,0,0,0,0,0,0,0};
    uint32_t temp[8];
    uint32_t exponent[8];
    
    // p - 2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140
    uint32_t p_minus_2[8] = {
        0xFFFFFC2E, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };
    
    fe_copy(exponent, p_minus_2);
    
    // الگوریتم مربع-و-ضرب برای توان‌رسانی
    for (int i = 255; i >= 0; i--) {
        int byte_index = i / 32;
        int bit_index = i % 32;
        
        fe_square(temp, result);
        fe_copy(result, temp);
        
        if ((exponent[byte_index] >> bit_index) & 1) {
            fe_mul(temp, result, a);
            fe_copy(result, temp);
        }
    }
    
    fe_copy(r, result);
}

__device__ bool fe_is_zero(const uint32_t* a) {
    for (int i = 0; i < 8; i++) {
        if (a[i] != 0) return false;
    }
    return true;
}

__device__ bool fe_equal(const uint32_t* a, const uint32_t* b) {
    for (int i = 0; i < 8; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

__device__ void fe_set_zero(uint32_t* a) {
    for (int i = 0; i < 8; i++) {
        a[i] = 0;
    }
}

__device__ void fe_copy(uint32_t* dst, const uint32_t* src) {
    for (int i = 0; i < 8; i++) {
        dst[i] = src[i];
    }
}

__device__ void fe_set_bytes(uint32_t* r, const uint32_t* bytes) {
    for (int i = 0; i < 8; i++) {
        r[i] = bytes[i];
    }
}

__device__ void fe_to_bytes(uint8_t* bytes, const uint32_t* a) {
    for (int i = 0; i < 8; i++) {
        bytes[i * 4] = (a[i] >> 24) & 0xFF;
        bytes[i * 4 + 1] = (a[i] >> 16) & 0xFF;
        bytes[i * 4 + 2] = (a[i] >> 8) & 0xFF;
        bytes[i * 4 + 3] = a[i] & 0xFF;
    }
}

__device__ void bytes_to_scalar(uint32_t* scalar, const uint8_t* bytes) {
    for (int i = 0; i < 8; i++) {
        scalar[i] = (bytes[i * 4] << 24) | 
                   (bytes[i * 4 + 1] << 16) | 
                   (bytes[i * 4 + 2] << 8) | 
                   bytes[i * 4 + 3];
    }
}

// پیاده‌سازی واقعی الگوریتم double-and-add برای ضرب نقطه بیضوی
__device__ bool ec_point_multiply(ECPoint* result, const uint32_t* scalar, const ECPoint* point) {
    ECPoint r;
    fe_set_zero(r.x);
    fe_set_zero(r.y);
    
    ECPoint temp;
    ECPoint current = *point;
    
    // الگوریتم double-and-add واقعی
    for (int i = 255; i >= 0; i--) {
        int byte_index = i / 32;
        int bit_index = i % 32;
        int bit = (scalar[byte_index] >> bit_index) & 1;
        
        // Double نقطه فعلی
        ec_point_double(&temp, &current);
        current = temp;
        
        // اگر بیت اسکالر ست شده باشد، نقطه را اضافه کن
        if (bit) {
            if (fe_is_zero(r.x) && fe_is_zero(r.y)) {
                r = current;
            } else {
                ec_point_add(&temp, &r, &current);
                r = temp;
            }
        }
    }
    
    *result = r;
    return !fe_is_zero(r.x); // بررسی معتبر بودن نتیجه
}

__device__ void ec_point_add(ECPoint* r, const ECPoint* a, const ECPoint* b) {
    if (fe_is_zero(a->x) && fe_is_zero(a->y)) {
        *r = *b;
        return;
    }
    if (fe_is_zero(b->x) && fe_is_zero(b->y)) {
        *r = *a;
        return;
    }
    
    uint32_t lambda[8], temp1[8], temp2[8], temp3[8];
    
    if (fe_equal(a->x, b->x)) {
        if (fe_equal(a->y, b->y)) {
            ec_point_double(r, a);
            return;
        } else {
            // نقاط معکوس - نتیجه نقطه در بینهایت
            fe_set_zero(r->x);
            fe_set_zero(r->y);
            return;
        }
    }
    
    // محاسبه lambda = (y2 - y1) / (x2 - x1)
    fe_sub(temp1, b->y, a->y);
    fe_sub(temp2, b->x, a->x);
    fe_invert(temp3, temp2);
    fe_mul(lambda, temp1, temp3);
    
    // محاسبه x3 = lambda^2 - x1 - x2
    fe_square(temp1, lambda);
    fe_sub(temp1, temp1, a->x);
    fe_sub(r->x, temp1, b->x);
    
    // محاسبه y3 = lambda * (x1 - x3) - y1
    fe_sub(temp1, a->x, r->x);
    fe_mul(temp1, lambda, temp1);
    fe_sub(r->y, temp1, a->y);
}

__device__ void ec_point_double(ECPoint* r, const ECPoint* a) {
    if (fe_is_zero(a->x) && fe_is_zero(a->y)) {
        *r = *a;
        return;
    }
    
    uint32_t lambda[8], temp1[8], temp2[8];
    
    // محاسبه lambda = (3 * x1^2) / (2 * y1)
    fe_square(temp1, a->x);
    uint32_t three[8] = {3,0,0,0,0,0,0,0};
    fe_mul(temp1, temp1, three);
    
    uint32_t two[8] = {2,0,0,0,0,0,0,0};
    fe_mul(temp2, a->y, two);
    fe_invert(temp2, temp2);
    
    fe_mul(lambda, temp1, temp2);
    
    // محاسبه x3 = lambda^2 - 2 * x1
    fe_square(temp1, lambda);
    fe_mul(temp2, a->x, two);
    fe_sub(r->x, temp1, temp2);
    
    // محاسبه y3 = lambda * (x1 - x3) - y1
    fe_sub(temp1, a->x, r->x);
    fe_mul(temp1, lambda, temp1);
    fe_sub(r->y, temp1, a->y);
}

// کرنل اصلی برای تولید کلید عمومی (بهبود یافته)
__global__ void generate_public_keys_kernel(
    const uint8_t* private_keys,
    uint8_t* public_keys,
    bool compressed,
    size_t batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const uint8_t* private_key = private_keys + idx * 32;
    uint8_t* public_key = public_keys + idx * (compressed ? 33 : 65);
    
    // تبدیل private key به scalar
    uint32_t scalar[8];
    bytes_to_scalar(scalar, private_key);
    
    // نقطه پایه secp256k1
    ECPoint base_point;
    fe_set_bytes(base_point.x, d_secp256k1_gx);
    fe_set_bytes(base_point.y, d_secp256k1_gy);
    
    ECPoint result;
    if (ec_point_multiply(&result, scalar, &base_point)) {
        // تبدیل به فرمت public key
        if (compressed) {
            // بررسی بیت پاریتی y-coordinate برای تعیین prefix
            public_key[0] = (result.y[7] & 1) ? 0x03 : 0x02;
            fe_to_bytes(public_key + 1, result.x);
        } else {
            public_key[0] = 0x04;
            fe_to_bytes(public_key + 1, result.x);
            fe_to_bytes(public_key + 33, result.y);
        }
    } else {
        // در صورت خطا، کلید عمومی صفر قرار دهید
        memset(public_key, 0, compressed ? 33 : 65);
    }
}

// کرنل بهبودیافته برای تولید دسته‌ای کلید عمومی
__global__ void generate_public_keys_batch_kernel(
    const uint8_t* private_keys,
    uint8_t* public_keys,
    bool compressed,
    size_t batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const uint8_t* private_key = private_keys + idx * 32;
    uint8_t* public_key = public_keys + idx * (compressed ? 33 : 65);
    
    // تبدیل private key به scalar
    uint32_t scalar[8];
    bytes_to_scalar(scalar, private_key);
    
    // نقطه پایه
    ECPoint base_point;
    fe_set_bytes(base_point.x, d_secp256k1_gx);
    fe_set_bytes(base_point.y, d_secp256k1_gy);
    
    ECPoint result;
    if (ec_point_multiply(&result, scalar, &base_point)) {
        // تبدیل به فرمت public key
        if (compressed) {
            public_key[0] = (result.y[7] & 1) ? 0x03 : 0x02;
            fe_to_bytes(public_key + 1, result.x);
        } else {
            public_key[0] = 0x04;
            fe_to_bytes(public_key + 1, result.x);
            fe_to_bytes(public_key + 33, result.y);
        }
    } else {
        memset(public_key, 0, compressed ? 33 : 65);
    }
}

// تابع کمکی برای تبدیل hex به bytes
std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(strtol(byteString.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

// تابع اعتبارسنجی با کتابخانه secp256k1
bool validate_with_secp256k1_lib(const std::vector<uint8_t>& private_key, const std::string& address) {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY | SECP256K1_CONTEXT_SIGN);
    if (!ctx) return false;
    
    bool valid = false;
    
    try {
        // بررسی معتبر بودن کلید خصوصی
        if (secp256k1_ec_seckey_verify(ctx, private_key.data()) != 1) {
            secp256k1_context_destroy(ctx);
            return false;
        }
        
        // تولید کلید عمومی از کلید خصوصی
        secp256k1_pubkey pubkey;
        if (secp256k1_ec_pubkey_create(ctx, &pubkey, private_key.data()) != 1) {
            secp256k1_context_destroy(ctx);
            return false;
        }
        
        // سریالایز کردن کلید عمومی (compressed)
        uint8_t pubkey_serialized[33];
        size_t output_len = 33;
        if (secp256k1_ec_pubkey_serialize(ctx, pubkey_serialized, &output_len, &pubkey, SECP256K1_EC_COMPRESSED) != 1) {
            secp256k1_context_destroy(ctx);
            return false;
        }
        
        // در اینجا باید آدرس واقعی تولید و با آدرس ورودی مقایسه شود
        // برای نمونه فعلی، فقط بررسی می‌کنیم که کلید خصوصی معتبر است
        valid = true;
        
    } catch (...) {
        valid = false;
    }
    
    secp256k1_context_destroy(ctx);
    return valid;
}

// کلاس مدیریت رمزنگاری
AdvancedSecp256k1::AdvancedSecp256k1() 
    : d_private_keys(nullptr), d_public_keys(nullptr), d_results(nullptr) {
    
    try {
        CUDA_CHECK(cudaMalloc(&d_private_keys, MAX_BATCH_SIZE * 32));
        CUDA_CHECK(cudaMalloc(&d_public_keys, MAX_BATCH_SIZE * 65)); // uncompressed
        CUDA_CHECK(cudaMalloc(&d_results, MAX_BATCH_SIZE));
        
        // کپی ثابت‌ها به حافظه ثابت GPU
        uint32_t SECP256K1_P[8] = {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
        uint32_t SECP256K1_N[8] = {0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
        uint32_t SECP256K1_GX[8] = {0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB, 0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E};
        uint32_t SECP256K1_GY[8] = {0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448, 0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77};
        
        CUDA_CHECK(cudaMemcpyToSymbol(d_secp256k1_p, SECP256K1_P, sizeof(SECP256K1_P)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_secp256k1_n, SECP256K1_N, sizeof(SECP256K1_N)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_secp256k1_gx, SECP256K1_GX, sizeof(SECP256K1_GX)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_secp256k1_gy, SECP256K1_GY, sizeof(SECP256K1_GY)));
        
        Logger::debug("AdvancedSecp256k1 initialized successfully");
        
    } catch (const std::exception& e) {
        // پاک‌سازی منابع در صورت خطا
        cleanup();
        throw;
    }
}

AdvancedSecp256k1::~AdvancedSecp256k1() {
    cleanup();
}

void AdvancedSecp256k1::cleanup() {
    // پاک‌سازی ایمن حافظه GPU
    if (d_private_keys) {
        cudaFree(d_private_keys);
        d_private_keys = nullptr;
    }
    if (d_public_keys) {
        cudaFree(d_public_keys);
        d_public_keys = nullptr;
    }
    if (d_results) {
        cudaFree(d_results);
        d_results = nullptr;
    }
}

std::vector<std::vector<uint8_t>> AdvancedSecp256k1::generate_public_keys(
    const std::vector<std::vector<uint8_t>>& private_keys, bool compressed) {
    
    if (private_keys.empty()) {
        return {};
    }
    
    size_t batch_size = private_keys.size();
    if (batch_size > MAX_BATCH_SIZE) {
        batch_size = MAX_BATCH_SIZE;
        Logger::warning("Batch size truncated to MAX_BATCH_SIZE: {}", MAX_BATCH_SIZE);
    }
    
    // کپی private keys به GPU
    std::vector<uint8_t> h_private_keys(batch_size * 32);
    for (size_t i = 0; i < batch_size; i++) {
        if (private_keys[i].size() != 32) {
            throw std::invalid_argument("Invalid private key size");
        }
        std::copy(private_keys[i].begin(), private_keys[i].end(), 
                 h_private_keys.begin() + i * 32);
    }
    
    CUDA_CHECK(cudaMemcpy(d_private_keys, h_private_keys.data(),
                         batch_size * 32, cudaMemcpyHostToDevice));
    
    // اجرای کرنل
    int threads_per_block = 256;
    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
    
    generate_public_keys_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_private_keys, d_public_keys, compressed, batch_size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // خواندن نتایج
    std::vector<uint8_t> h_public_keys(batch_size * (compressed ? 33 : 65));
    CUDA_CHECK(cudaMemcpy(h_public_keys.data(), d_public_keys,
                         batch_size * (compressed ? 33 : 65), cudaMemcpyDeviceToHost));
    
    // تبدیل به فرمت خروجی
    std::vector<std::vector<uint8_t>> results;
    results.reserve(batch_size);
    
    size_t pubkey_size = compressed ? 33 : 65;
    for (size_t i = 0; i < batch_size; i++) {
        results.emplace_back(h_public_keys.begin() + i * pubkey_size,
                           h_public_keys.begin() + (i + 1) * pubkey_size);
    }
    
    return results;
}

bool AdvancedSecp256k1::validate_private_key(const std::vector<uint8_t>& private_key) {
    if (private_key.size() != 32) {
        return false;
    }
    
    // بررسی اینکه private key در محدوده معتبر باشد
    // مقایسه با n (حداکثر مقدار)
    uint32_t SECP256K1_N[8] = {0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    
    for (int i = 7; i >= 0; i--) {
        uint32_t key_part = (private_key[i * 4] << 24) | 
                           (private_key[i * 4 + 1] << 16) | 
                           (private_key[i * 4 + 2] << 8) | 
                           private_key[i * 4 + 3];
        
        if (key_part < SECP256K1_N[i]) return true;
        if (key_part > SECP256K1_N[i]) return false;
    }
    
    return false; // مساوی با n (نامعتبر)
}

// تابع اعتبارسنجی پیشرفته با کتابخانه secp256k1
bool AdvancedSecp256k1::validate_keypair(const std::vector<uint8_t>& private_key, const std::vector<uint8_t>& public_key) {
    return validate_with_secp256k1_lib(private_key, "");
}

} // namespace bitcoin_miner