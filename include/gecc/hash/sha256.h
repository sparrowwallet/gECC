#pragma once

#include <cstdint>
#include <cstring>

namespace gecc {
namespace hash {

// SHA-256 constants (first 32 bits of fractional parts of cube roots of first 64 primes)
__device__ __constant__ static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
struct SHA256State {
    uint32_t h[8];

    __device__ __host__ __forceinline__ void init() {
        h[0] = 0x6a09e667;
        h[1] = 0xbb67ae85;
        h[2] = 0x3c6ef372;
        h[3] = 0xa54ff53a;
        h[4] = 0x510e527f;
        h[5] = 0x9b05688c;
        h[6] = 0x1f83d9ab;
        h[7] = 0x5be0cd19;
    }
};

// Bitwise rotate right
__device__ __host__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 functions
__device__ __host__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __host__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __host__ __forceinline__ uint32_t Sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __host__ __forceinline__ uint32_t Sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __host__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __host__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Process a single 512-bit block
__device__ __forceinline__ void sha256_process_block_device(SHA256State& state, const uint8_t* block) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Prepare message schedule (big-endian)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }

    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }

    // Initialize working variables
    a = state.h[0];
    b = state.h[1];
    c = state.h[2];
    d = state.h[3];
    e = state.h[4];
    f = state.h[5];
    g = state.h[6];
    h = state.h[7];

    // Main compression loop
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t T2 = Sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Add compressed chunk to current hash value
    state.h[0] += a;
    state.h[1] += b;
    state.h[2] += c;
    state.h[3] += d;
    state.h[4] += e;
    state.h[5] += f;
    state.h[6] += g;
    state.h[7] += h;
}

// Host version using local K array
__host__ __forceinline__ void sha256_process_block_host(SHA256State& state, const uint8_t* block) {
    // Local copy of K constants for host code
    static const uint32_t K_host[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Prepare message schedule (big-endian)
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }

    for (int i = 16; i < 64; ++i) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }

    // Initialize working variables
    a = state.h[0];
    b = state.h[1];
    c = state.h[2];
    d = state.h[3];
    e = state.h[4];
    f = state.h[5];
    g = state.h[6];
    h = state.h[7];

    // Main compression loop
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K_host[i] + W[i];
        uint32_t T2 = Sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Add compressed chunk to current hash value
    state.h[0] += a;
    state.h[1] += b;
    state.h[2] += c;
    state.h[3] += d;
    state.h[4] += e;
    state.h[5] += f;
    state.h[6] += g;
    state.h[7] += h;
}

// SHA-256 hash function - device version
__device__ __forceinline__ void sha256(const uint8_t* data, uint64_t len, uint8_t* hash) {
    SHA256State state;
    state.init();

    uint64_t num_blocks = (len + 8) / 64 + 1; // Including padding and length
    uint64_t padded_len = num_blocks * 64;

    uint8_t block[64];

    for (uint64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint64_t block_start = block_idx * 64;

        // Copy data or add padding
        for (int i = 0; i < 64; ++i) {
            uint64_t pos = block_start + i;
            if (pos < len) {
                block[i] = data[pos];
            } else if (pos == len) {
                block[i] = 0x80; // Append '1' bit followed by zeros
            } else if (pos < padded_len - 8) {
                block[i] = 0x00;
            } else {
                // Append length in bits as 64-bit big-endian
                int len_byte_idx = (int)(pos - (padded_len - 8));
                block[i] = (uint8_t)((len * 8) >> (56 - len_byte_idx * 8));
            }
        }

        sha256_process_block_device(state, block);
    }

    // Output hash (big-endian)
    for (int i = 0; i < 8; ++i) {
        hash[i * 4] = (uint8_t)(state.h[i] >> 24);
        hash[i * 4 + 1] = (uint8_t)(state.h[i] >> 16);
        hash[i * 4 + 2] = (uint8_t)(state.h[i] >> 8);
        hash[i * 4 + 3] = (uint8_t)(state.h[i]);
    }
}

// SHA-256 hash function - host version
__host__ __forceinline__ void sha256_host(const uint8_t* data, uint64_t len, uint8_t* hash) {
    SHA256State state;
    state.init();

    uint64_t num_blocks = (len + 8) / 64 + 1; // Including padding and length
    uint64_t padded_len = num_blocks * 64;

    uint8_t block[64];

    for (uint64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint64_t block_start = block_idx * 64;

        // Copy data or add padding
        for (int i = 0; i < 64; ++i) {
            uint64_t pos = block_start + i;
            if (pos < len) {
                block[i] = data[pos];
            } else if (pos == len) {
                block[i] = 0x80; // Append '1' bit followed by zeros
            } else if (pos < padded_len - 8) {
                block[i] = 0x00;
            } else {
                // Append length in bits as 64-bit big-endian
                int len_byte_idx = (int)(pos - (padded_len - 8));
                block[i] = (uint8_t)((len * 8) >> (56 - len_byte_idx * 8));
            }
        }

        sha256_process_block_host(state, block);
    }

    // Output hash (big-endian)
    for (int i = 0; i < 8; ++i) {
        hash[i * 4] = (uint8_t)(state.h[i] >> 24);
        hash[i * 4 + 1] = (uint8_t)(state.h[i] >> 16);
        hash[i * 4 + 2] = (uint8_t)(state.h[i] >> 8);
        hash[i * 4 + 3] = (uint8_t)(state.h[i]);
    }
}

// Tagged hash function - device version
// Computes: SHA256(SHA256(tag) || SHA256(tag) || msg)
// This is used in BIP-340 and other Bitcoin protocols
__device__ __forceinline__ void tagged_hash(const uint8_t* tag, uint64_t tag_len,
                                             const uint8_t* msg, uint64_t msg_len,
                                             uint8_t* hash) {
    // Compute tag_hash = SHA256(tag)
    uint8_t tag_hash[32];
    sha256(tag, tag_len, tag_hash);

    // Allocate buffer for: tag_hash || tag_hash || msg
    // Note: In practice, for large messages, this could be optimized to avoid full buffer allocation
    // by using incremental SHA-256 state updates
    const uint64_t total_len = 32 + 32 + msg_len;

    // For small messages (< 960 bytes total), use stack allocation
    if (total_len <= 960) {
        uint8_t buffer[960];

        // Copy: tag_hash || tag_hash || msg
        for (int i = 0; i < 32; ++i) {
            buffer[i] = tag_hash[i];
            buffer[32 + i] = tag_hash[i];
        }
        for (uint64_t i = 0; i < msg_len; ++i) {
            buffer[64 + i] = msg[i];
        }

        // Compute final hash
        sha256(buffer, total_len, hash);
    } else {
        // For larger messages, we'd need a different approach (incremental hashing)
        // For now, this is a limitation - most use cases have small messages
        // Could be extended with dynamic allocation or incremental API
    }
}

// Tagged hash function - host version
__host__ __forceinline__ void tagged_hash_host(const uint8_t* tag, uint64_t tag_len,
                                                const uint8_t* msg, uint64_t msg_len,
                                                uint8_t* hash) {
    // Compute tag_hash = SHA256(tag)
    uint8_t tag_hash[32];
    sha256_host(tag, tag_len, tag_hash);

    // Allocate buffer for: tag_hash || tag_hash || msg
    const uint64_t total_len = 32 + 32 + msg_len;
    uint8_t* buffer = new uint8_t[total_len];

    // Copy: tag_hash || tag_hash || msg
    memcpy(buffer, tag_hash, 32);
    memcpy(buffer + 32, tag_hash, 32);
    memcpy(buffer + 64, msg, msg_len);

    // Compute final hash
    sha256_host(buffer, total_len, hash);

    delete[] buffer;
}

// Batch SHA-256 for EC point outputs (64 bytes: 32-byte x || 32-byte y)
// Each thread hashes one EC point
__global__ void batch_sha256_ec_points(
    const uint8_t* points,  // Input: points in row-major format (64 bytes per point)
    uint8_t* hashes,        // Output: 32-byte hashes
    uint32_t count          // Number of points
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        const uint8_t* point = points + idx * 64;
        uint8_t* hash = hashes + idx * 32;
        sha256(point, 64, hash);
    }
}

// Batch tagged hash
// Each thread computes: SHA256(SHA256(tag) || SHA256(tag) || msg[i])
// tag is shared across all threads, msgs are per-thread
__global__ void batch_tagged_hash(
    const uint8_t* tag,      // Input: tag string (shared by all threads)
    uint32_t tag_len,        // Length of tag
    const uint8_t* msgs,     // Input: messages in row-major format
    const uint32_t* msg_lens, // Input: length of each message
    uint8_t* hashes,         // Output: 32-byte hashes
    uint32_t count           // Number of messages
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        // Compute offset for this message (assuming fixed-size messages for simplicity)
        // For variable-length messages, would need a prefix sum of msg_lens
        uint32_t msg_len = msg_lens[idx];

        // Note: This assumes msgs are stored contiguously with max_msg_len padding
        // For true variable-length, would need offset array
        const uint8_t* msg = msgs + idx * msg_len;
        uint8_t* hash = hashes + idx * 32;

        tagged_hash(tag, tag_len, msg, msg_len, hash);
    }
}

} // namespace hash
} // namespace gecc
