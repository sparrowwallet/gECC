#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstring>
#include <iomanip>
#include <sstream>
#include "gecc/hash/sha256.h"
#include "sha256_test_constants.h"

using namespace gecc::hash;

// Helper function to convert bytes to hex string
std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << (int)data[i];
    }
    return ss.str();
}

// Helper function to convert hex string to bytes
void hex_to_bytes(const char* hex, uint8_t* bytes, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        sscanf(hex + 2 * i, "%2hhx", &bytes[i]);
    }
}

// Test SHA-256 on CPU with standard test vectors
TEST(SHA256Test, StandardVectors) {
    for (int i = 0; i < SHA256_NUM_STANDARD_TESTS; ++i) {
        uint8_t hash[32];
        sha256_host(SHA256_STANDARD_INPUTS[i], SHA256_STANDARD_INPUT_LENS[i], hash);

        std::string computed = bytes_to_hex(hash, 32);
        std::string expected = SHA256_STANDARD_EXPECTED[i];

        EXPECT_EQ(computed, expected)
            << "Test " << i << " failed\n"
            << "Input: \"" << SHA256_STANDARD_INPUTS[i] << "\"\n"
            << "Expected: " << expected << "\n"
            << "Computed: " << computed;
    }
}

// Kernel for GPU standard test vectors
__global__ void sha256_standard_test_kernel(uint8_t** inputs, uint64_t* lens, uint8_t* hashes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        sha256(inputs[idx], lens[idx], hashes + idx * 32);
    }
}

// Test SHA-256 on GPU with standard test vectors
TEST(SHA256Test, GPUStandardVectors) {
    const int num_tests = SHA256_NUM_STANDARD_TESTS;

    // Allocate device memory
    uint8_t** d_inputs;
    uint8_t* d_hashes;
    cudaMalloc(&d_inputs, num_tests * sizeof(uint8_t*));
    cudaMalloc(&d_hashes, num_tests * 32);

    // Copy inputs to device
    uint8_t* h_input_ptrs[num_tests];
    for (int i = 0; i < num_tests; ++i) {
        cudaMalloc(&h_input_ptrs[i], SHA256_STANDARD_INPUT_LENS[i] + 1);
        cudaMemcpy(h_input_ptrs[i], SHA256_STANDARD_INPUTS[i],
                   SHA256_STANDARD_INPUT_LENS[i], cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_inputs, h_input_ptrs, num_tests * sizeof(uint8_t*), cudaMemcpyHostToDevice);

    // Copy lengths to device
    uint64_t* d_lens;
    cudaMalloc(&d_lens, num_tests * sizeof(uint64_t));
    cudaMemcpy(d_lens, SHA256_STANDARD_INPUT_LENS, num_tests * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    sha256_standard_test_kernel<<<(num_tests + 31) / 32, 32>>>(d_inputs, d_lens, d_hashes, num_tests);
    cudaDeviceSynchronize();

    // Copy results back
    uint8_t h_hashes[num_tests * 32];
    cudaMemcpy(h_hashes, d_hashes, num_tests * 32, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < num_tests; ++i) {
        std::string computed = bytes_to_hex(h_hashes + i * 32, 32);
        std::string expected = SHA256_STANDARD_EXPECTED[i];

        EXPECT_EQ(computed, expected)
            << "GPU Test " << i << " failed\n"
            << "Input: \"" << SHA256_STANDARD_INPUTS[i] << "\"\n"
            << "Expected: " << expected << "\n"
            << "Computed: " << computed;
    }

    // Cleanup
    for (int i = 0; i < num_tests; ++i) {
        cudaFree(h_input_ptrs[i]);
    }
    cudaFree(d_inputs);
    cudaFree(d_lens);
    cudaFree(d_hashes);
}

// Test batch SHA-256 for 64-byte EC point data
TEST(SHA256Test, BatchECPoints) {
    const int num_points = SHA256_EC_NUM_TESTS;
    const int point_size = 64;
    const int hash_size = 32;

    // Allocate host memory
    uint8_t* h_points = new uint8_t[num_points * point_size];
    uint8_t* h_hashes = new uint8_t[num_points * hash_size];

    // Copy test points
    for (int i = 0; i < num_points; ++i) {
        memcpy(h_points + i * point_size, SHA256_EC_POINTS[i], point_size);
    }

    // Allocate device memory
    uint8_t *d_points, *d_hashes;
    cudaMalloc(&d_points, num_points * point_size);
    cudaMalloc(&d_hashes, num_points * hash_size);

    // Copy to device
    cudaMemcpy(d_points, h_points, num_points * point_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;
    batch_sha256_ec_points<<<num_blocks, threads_per_block>>>(d_points, d_hashes, num_points);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_hashes, d_hashes, num_points * hash_size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < num_points; ++i) {
        std::string computed = bytes_to_hex(h_hashes + i * hash_size, hash_size);
        std::string expected = SHA256_EC_EXPECTED[i];

        EXPECT_EQ(computed, expected)
            << "EC Point Test " << i << " failed\n"
            << "Expected: " << expected << "\n"
            << "Computed: " << computed;
    }

    // Cleanup
    delete[] h_points;
    delete[] h_hashes;
    cudaFree(d_points);
    cudaFree(d_hashes);
}

// Test tagged hash on CPU
TEST(SHA256Test, TaggedHashCPU) {
    // Test case from BIP-340 specification
    const char* tag = "BIP0340/challenge";
    const uint8_t msg[] = {0x01, 0x02, 0x03, 0x04};
    uint8_t hash[32];

    tagged_hash_host((const uint8_t*)tag, strlen(tag), msg, sizeof(msg), hash);

    // Verify it's different from plain SHA-256
    uint8_t plain_hash[32];
    sha256_host(msg, sizeof(msg), plain_hash);

    bool different = false;
    for (int i = 0; i < 32; ++i) {
        if (hash[i] != plain_hash[i]) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "Tagged hash should differ from plain SHA-256";

    // Manually verify the construction: SHA256(SHA256(tag) || SHA256(tag) || msg)
    uint8_t tag_hash[32];
    sha256_host((const uint8_t*)tag, strlen(tag), tag_hash);

    size_t total_len = 32 + 32 + sizeof(msg);
    uint8_t* buffer = new uint8_t[total_len];
    memcpy(buffer, tag_hash, 32);
    memcpy(buffer + 32, tag_hash, 32);
    memcpy(buffer + 64, msg, sizeof(msg));

    uint8_t expected[32];
    sha256_host(buffer, total_len, expected);

    std::string computed = bytes_to_hex(hash, 32);
    std::string expected_str = bytes_to_hex(expected, 32);

    EXPECT_EQ(computed, expected_str)
        << "Tagged hash should match manual construction";

    delete[] buffer;
}

// Kernel for GPU tagged hash test
__global__ void tagged_hash_test_kernel(
    const uint8_t* tag, uint32_t tag_len,
    const uint8_t* msg, uint32_t msg_len,
    uint8_t* hash
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        tagged_hash(tag, tag_len, msg, msg_len, hash);
    }
}

// Test tagged hash on GPU
TEST(SHA256Test, TaggedHashGPU) {
    const char* tag = "BIP0340/challenge";
    const uint8_t msg[] = {0x01, 0x02, 0x03, 0x04};

    // Allocate device memory
    uint8_t *d_tag, *d_msg, *d_hash;
    cudaMalloc(&d_tag, strlen(tag));
    cudaMalloc(&d_msg, sizeof(msg));
    cudaMalloc(&d_hash, 32);

    // Copy to device
    cudaMemcpy(d_tag, tag, strlen(tag), cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg, msg, sizeof(msg), cudaMemcpyHostToDevice);

    // Launch kernel
    tagged_hash_test_kernel<<<1, 1>>>(d_tag, strlen(tag), d_msg, sizeof(msg), d_hash);
    cudaDeviceSynchronize();

    // Copy result back
    uint8_t h_hash[32];
    cudaMemcpy(h_hash, d_hash, 32, cudaMemcpyDeviceToHost);

    // Compute expected on CPU
    uint8_t expected[32];
    tagged_hash_host((const uint8_t*)tag, strlen(tag), msg, sizeof(msg), expected);

    std::string computed = bytes_to_hex(h_hash, 32);
    std::string expected_str = bytes_to_hex(expected, 32);

    EXPECT_EQ(computed, expected_str)
        << "GPU tagged hash should match CPU version";

    // Cleanup
    cudaFree(d_tag);
    cudaFree(d_msg);
    cudaFree(d_hash);
}
