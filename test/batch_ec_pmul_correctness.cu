#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "gecc.h"
#include "gecc/arith/layout.h"
#include "gecc/arith/ec.h"

using namespace gecc;
using namespace arith;

#include "batch_pmul_test_constants.h"

static_assert(MAX_BYTES >= 16 * 8, "");

// Define field and EC types for secp256k1
// secp256k1 has a=0, so use DBL_FLAG=1 for the a=0 optimized doubling formula
DEFINE_SECP256K1_FP(Fq_SECP256K1, FqSECP256K1, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::DEFAULT);
DEFINE_EC(G1_EC, G1SECP256K1, Fq_SECP256K1, SECP256K1_CURVE, 1);

using ECPoint = G1_EC_G1SECP256K1;
using Field = Fq_SECP256K1;
using AffinePoint = ECPoint::Affine;

// Kernel to perform batch scalar multiplication
// Each thread computes one scalar multiplication: result[i] = scalar[i] * point[i]
__global__ void BatchScalarMultiplication(
    const uint32_t *input_points_x,  // X coordinates of input points (column-major if GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS)
    const uint32_t *input_points_y,  // Y coordinates of input points
    const uint32_t *scalars,         // Scalars to multiply by
    uint32_t *output_points_x,       // X coordinates of results
    uint32_t *output_points_y,       // Y coordinates of results
    uint32_t count) {                // Number of points to process

  const u32 slot_idx = LayoutT<1>::global_slot_idx();
  const u32 lane_idx = LayoutT<1>::lane_idx();

  if (slot_idx >= count) return;

  // Load input point
  Field px, py;
  px.load_arbitrary(input_points_x, count, slot_idx, lane_idx);
  py.load_arbitrary(input_points_y, count, slot_idx, lane_idx);

  // Convert to Montgomery form
  px.inplace_to_montgomery();
  py.inplace_to_montgomery();

  // Create affine point
  AffinePoint base_point{px, py};

  // Load scalar
  Field scalar;
  scalar.load_arbitrary(scalars, count, slot_idx, lane_idx);

  // Convert to Jacobian for computation
  ECPoint result_jac = ECPoint::zero();
  ECPoint base_jac = base_point.to_jacobian();

  // Double-and-add algorithm for scalar multiplication
  for (int bit = Field::BITS - 1; bit >= 0; --bit) {
    result_jac = result_jac.dbl();

    // Check if bit is set
    u32 limb_idx = bit / 32;
    u32 bit_pos = bit % 32;
    bool bit_set = (scalar.digits[limb_idx] >> bit_pos) & 1;

    if (bit_set) {
      result_jac = result_jac + base_jac;
    }
  }

  // Convert result to affine
  AffinePoint result = result_jac.to_affine();

  // Convert from Montgomery form
  Field result_x = result.x.from_montgomery();
  Field result_y = result.y.from_montgomery();

  // Store result
  result_x.store_arbitrary(output_points_x, count, slot_idx, lane_idx);
  result_y.store_arbitrary(output_points_y, count, slot_idx, lane_idx);
}

// Helper function to convert from uint64_t limbs to uint32_t limbs
void convert_u64_to_u32_limbs(const uint64_t *input, uint32_t *output, size_t num_u64_limbs) {
  for (size_t i = 0; i < num_u64_limbs; ++i) {
    output[2*i] = static_cast<uint32_t>(input[i] & 0xFFFFFFFFULL);
    output[2*i + 1] = static_cast<uint32_t>(input[i] >> 32);
  }
}

// Helper function to convert from uint32_t limbs to uint64_t limbs
void convert_u32_to_u64_limbs(const uint32_t *input, uint64_t *output, size_t num_u64_limbs) {
  for (size_t i = 0; i < num_u64_limbs; ++i) {
    output[i] = static_cast<uint64_t>(input[2*i]) |
                (static_cast<uint64_t>(input[2*i + 1]) << 32);
  }
}

// Test batch scalar multiplication correctness
TEST(BatchECOperations, ScalarMultiplicationCorrectness) {
  Fq_SECP256K1::initialize();

  const u32 num_tests = BATCH_PMUL_NUM_TESTS;
  const u32 limbs_per_field = 8; // u32 limbs for 256-bit field

  // Allocate host memory for test data (in u32 format)
  std::vector<uint32_t> h_points_x(num_tests * limbs_per_field);
  std::vector<uint32_t> h_points_y(num_tests * limbs_per_field);
  std::vector<uint32_t> h_scalars(num_tests * limbs_per_field);
  std::vector<uint32_t> h_expected_x(num_tests * limbs_per_field);
  std::vector<uint32_t> h_expected_y(num_tests * limbs_per_field);

  // Convert test data from uint64_t to uint32_t format
  for (u32 i = 0; i < num_tests; ++i) {
    convert_u64_to_u32_limbs(&BATCH_PMUL_POINTS_X[i][0], &h_points_x[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_PMUL_POINTS_Y[i][0], &h_points_y[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_PMUL_SCALARS[i][0], &h_scalars[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_PMUL_EXPECTED_X[i][0], &h_expected_x[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_PMUL_EXPECTED_Y[i][0], &h_expected_y[i * limbs_per_field], 4);
  }

#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
  // Transform to column-major layout
  std::vector<uint32_t> h_points_x_col(num_tests * limbs_per_field);
  std::vector<uint32_t> h_points_y_col(num_tests * limbs_per_field);
  std::vector<uint32_t> h_scalars_col(num_tests * limbs_per_field);

  for (u32 j = 0; j < limbs_per_field; ++j) {
    for (u32 i = 0; i < num_tests; ++i) {
      h_points_x_col[j * num_tests + i] = h_points_x[i * limbs_per_field + j];
      h_points_y_col[j * num_tests + i] = h_points_y[i * limbs_per_field + j];
      h_scalars_col[j * num_tests + i] = h_scalars[i * limbs_per_field + j];
    }
  }

  // Allocate device memory
  uint32_t *d_points_x, *d_points_y, *d_scalars;
  uint32_t *d_results_x, *d_results_y;

  cudaMalloc(&d_points_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_points_y, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_scalars, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_results_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_results_y, num_tests * limbs_per_field * sizeof(uint32_t));

  // Copy column-major data to device
  cudaMemcpy(d_points_x, h_points_x_col.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points_y, h_points_y_col.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scalars, h_scalars_col.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
#else
  // Allocate device memory
  uint32_t *d_points_x, *d_points_y, *d_scalars;
  uint32_t *d_results_x, *d_results_y;

  cudaMalloc(&d_points_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_points_y, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_scalars, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_results_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_results_y, num_tests * limbs_per_field * sizeof(uint32_t));

  // Copy row-major data to device
  cudaMemcpy(d_points_x, h_points_x.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points_y, h_points_y.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scalars, h_scalars.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
#endif

  // Launch kernel
  u32 threads_per_block = 256;
  u32 num_blocks = (num_tests + threads_per_block - 1) / threads_per_block;

  BatchScalarMultiplication<<<num_blocks, threads_per_block>>>(
    d_points_x, d_points_y, d_scalars,
    d_results_x, d_results_y, num_tests
  );

  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint32_t> h_results_x(num_tests * limbs_per_field);
  std::vector<uint32_t> h_results_y(num_tests * limbs_per_field);

#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
  std::vector<uint32_t> h_results_x_col(num_tests * limbs_per_field);
  std::vector<uint32_t> h_results_y_col(num_tests * limbs_per_field);

  cudaMemcpy(h_results_x_col.data(), d_results_x, num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_results_y_col.data(), d_results_y, num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // Transform back from column-major to row-major
  for (u32 j = 0; j < limbs_per_field; ++j) {
    for (u32 i = 0; i < num_tests; ++i) {
      h_results_x[i * limbs_per_field + j] = h_results_x_col[j * num_tests + i];
      h_results_y[i * limbs_per_field + j] = h_results_y_col[j * num_tests + i];
    }
  }
#else
  cudaMemcpy(h_results_x.data(), d_results_x, num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_results_y.data(), d_results_y, num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyDeviceToHost);
#endif

  // Verify results
  u32 num_correct = 0;
  for (u32 i = 0; i < num_tests; ++i) {
    bool x_matches = true, y_matches = true;

    for (u32 j = 0; j < limbs_per_field; ++j) {
      if (h_results_x[i * limbs_per_field + j] != h_expected_x[i * limbs_per_field + j]) {
        x_matches = false;
      }
      if (h_results_y[i * limbs_per_field + j] != h_expected_y[i * limbs_per_field + j]) {
        y_matches = false;
      }
    }

    if (x_matches && y_matches) {
      num_correct++;
    } else {
      // Print first few failures for debugging
      if (num_correct < 3) {
        printf("Test %u failed:\n", i);
        printf("  Result X:   ");
        for (int j = limbs_per_field - 1; j >= 0; --j) {
          printf("%08x ", h_results_x[i * limbs_per_field + j]);
        }
        printf("\n  Expected X: ");
        for (int j = limbs_per_field - 1; j >= 0; --j) {
          printf("%08x ", h_expected_x[i * limbs_per_field + j]);
        }
        printf("\n");
      }
    }
  }

  // Cleanup
  cudaFree(d_points_x);
  cudaFree(d_points_y);
  cudaFree(d_scalars);
  cudaFree(d_results_x);
  cudaFree(d_results_y);

  ASSERT_EQ(num_correct, num_tests) << "Only " << num_correct << " out of " << num_tests << " tests passed";
}
