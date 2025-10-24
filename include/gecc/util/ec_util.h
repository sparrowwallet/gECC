#pragma once

#include <cstdint>

namespace gecc {
namespace util {

// Extract a big-endian int64 from the first 8 bytes of data
// Typically used on the x-coordinate of an EC point
__device__ __host__ __forceinline__ int64_t extract_bigendian_int64(const uint8_t* data) {
    int64_t result = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result = (result << 8) | data[i];
    }
    return result;
}

// Batch extraction of int64 values from EC point x-coordinates
// Each thread extracts one int64 from one EC point's x-coordinate
// Input: points_x - EC point x-coordinates in row-major format (32 bytes each)
// Output: values - extracted int64 values
__global__ void batch_extract_int64_from_ec_x(
    const uint8_t* points_x,  // Input: x-coordinates in big-endian bytes (32 bytes per point)
    int64_t* values,          // Output: extracted int64 values
    uint32_t count            // Number of points
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        const uint8_t* x_coord = points_x + idx * 32;
        values[idx] = extract_bigendian_int64(x_coord);
    }
}

// Extract int64 from field element (u32 limbs in Montgomery form)
// Converts from Montgomery form, then extracts first 8 bytes as big-endian int64
// This is for working directly with GPU EC arithmetic output
template<typename Field>
__device__ __forceinline__ int64_t extract_int64_from_field(const Field& x) {
    // Convert from Montgomery form to normal form
    Field x_normal = x.from_montgomery();

    // Convert to big-endian bytes
    // Field uses u32 limbs in little-endian order (limb 0 is LSB)
    // We need the most significant 8 bytes
    uint8_t bytes[8];

    // Extract top 8 bytes (assuming 256-bit field = 8 u32 limbs)
    // Limbs 6 and 7 contain the most significant 64 bits
    static_assert(Field::LIMBS_PER_LANE >= 8, "Field must be at least 256 bits");

    // Limb 7 (most significant 32 bits) -> bytes[0..3]
    bytes[0] = (uint8_t)(x_normal.digits[7] >> 24);
    bytes[1] = (uint8_t)(x_normal.digits[7] >> 16);
    bytes[2] = (uint8_t)(x_normal.digits[7] >> 8);
    bytes[3] = (uint8_t)(x_normal.digits[7]);

    // Limb 6 (next 32 bits) -> bytes[4..7]
    bytes[4] = (uint8_t)(x_normal.digits[6] >> 24);
    bytes[5] = (uint8_t)(x_normal.digits[6] >> 16);
    bytes[6] = (uint8_t)(x_normal.digits[6] >> 8);
    bytes[7] = (uint8_t)(x_normal.digits[6]);

    // Combine into int64
    return extract_bigendian_int64(bytes);
}

} // namespace util
} // namespace gecc
