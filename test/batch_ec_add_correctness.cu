#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "gecc.h"
#include "gecc/arith/layout.h"
#include "gecc/arith/ec.h"

using namespace gecc;
using namespace arith;

#include "batch_add_test_constants.h"

static_assert(MAX_BYTES >= 16 * 8, "");

// Define field and EC types for secp256k1
// secp256k1 has a=0, so use DBL_FLAG=1 for the a=0 optimized doubling formula
DEFINE_SECP256K1_FP(Fq_SECP256K1, FqSECP256K1, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::DEFAULT);
DEFINE_EC(G1_EC, G1SECP256K1, Fq_SECP256K1, SECP256K1_CURVE, 1);

using ECPoint = G1_EC_G1SECP256K1;
using Field = Fq_SECP256K1;
using AffinePoint = ECPoint::Affine;

// Kernel to perform batch point addition
// Each thread computes one point addition: result[i] = point1[i] + point2[i]
__global__ void BatchPointAddition(
    const uint32_t *points1_x,       // X coordinates of first points (column-major if GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS)
    const uint32_t *points1_y,       // Y coordinates of first points
    const uint32_t *points2_x,       // X coordinates of second points
    const uint32_t *points2_y,       // Y coordinates of second points
    uint32_t *output_points_x,       // X coordinates of results
    uint32_t *output_points_y,       // Y coordinates of results
    uint32_t count) {                // Number of additions to perform

  const u32 slot_idx = LayoutT<1>::global_slot_idx();
  const u32 lane_idx = LayoutT<1>::lane_idx();

  if (slot_idx >= count) return;

  // Load first point
  Field p1x, p1y;
  p1x.load_arbitrary(points1_x, count, slot_idx, lane_idx);
  p1y.load_arbitrary(points1_y, count, slot_idx, lane_idx);

  // Convert to Montgomery form
  p1x.inplace_to_montgomery();
  p1y.inplace_to_montgomery();

  // Create affine point
  AffinePoint point1{p1x, p1y};

  // Load second point
  Field p2x, p2y;
  p2x.load_arbitrary(points2_x, count, slot_idx, lane_idx);
  p2y.load_arbitrary(points2_y, count, slot_idx, lane_idx);

  // Convert to Montgomery form
  p2x.inplace_to_montgomery();
  p2y.inplace_to_montgomery();

  // Create affine point
  AffinePoint point2{p2x, p2y};

  // Convert to Jacobian for computation
  ECPoint p1_jac = point1.to_jacobian();
  ECPoint p2_jac = point2.to_jacobian();

  // Perform addition
  ECPoint result_jac = p1_jac + p2_jac;

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

// Test batch point addition correctness
TEST(BatchECOperations, PointAdditionCorrectness) {
  Fq_SECP256K1::initialize();

  const u32 num_tests = BATCH_ADD_NUM_TESTS;
  const u32 limbs_per_field = 8; // u32 limbs for 256-bit field

  // Allocate host memory for test data (in u32 format)
  std::vector<uint32_t> h_points1_x(num_tests * limbs_per_field);
  std::vector<uint32_t> h_points1_y(num_tests * limbs_per_field);
  std::vector<uint32_t> h_points2_x(num_tests * limbs_per_field);
  std::vector<uint32_t> h_points2_y(num_tests * limbs_per_field);
  std::vector<uint32_t> h_expected_x(num_tests * limbs_per_field);
  std::vector<uint32_t> h_expected_y(num_tests * limbs_per_field);

  // Convert test data from uint64_t to uint32_t format
  for (u32 i = 0; i < num_tests; ++i) {
    convert_u64_to_u32_limbs(&BATCH_ADD_POINTS1_X[i][0], &h_points1_x[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_ADD_POINTS1_Y[i][0], &h_points1_y[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_ADD_POINTS2_X[i][0], &h_points2_x[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_ADD_POINTS2_Y[i][0], &h_points2_y[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_ADD_EXPECTED_X[i][0], &h_expected_x[i * limbs_per_field], 4);
    convert_u64_to_u32_limbs(&BATCH_ADD_EXPECTED_Y[i][0], &h_expected_y[i * limbs_per_field], 4);
  }

#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
  // Transform to column-major layout
  std::vector<uint32_t> h_points1_x_col(num_tests * limbs_per_field);
  std::vector<uint32_t> h_points1_y_col(num_tests * limbs_per_field);
  std::vector<uint32_t> h_points2_x_col(num_tests * limbs_per_field);
  std::vector<uint32_t> h_points2_y_col(num_tests * limbs_per_field);

  for (u32 j = 0; j < limbs_per_field; ++j) {
    for (u32 i = 0; i < num_tests; ++i) {
      h_points1_x_col[j * num_tests + i] = h_points1_x[i * limbs_per_field + j];
      h_points1_y_col[j * num_tests + i] = h_points1_y[i * limbs_per_field + j];
      h_points2_x_col[j * num_tests + i] = h_points2_x[i * limbs_per_field + j];
      h_points2_y_col[j * num_tests + i] = h_points2_y[i * limbs_per_field + j];
    }
  }

  // Allocate device memory
  uint32_t *d_points1_x, *d_points1_y, *d_points2_x, *d_points2_y;
  uint32_t *d_results_x, *d_results_y;

  cudaMalloc(&d_points1_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_points1_y, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_points2_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_points2_y, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_results_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_results_y, num_tests * limbs_per_field * sizeof(uint32_t));

  // Copy column-major data to device
  cudaMemcpy(d_points1_x, h_points1_x_col.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points1_y, h_points1_y_col.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points2_x, h_points2_x_col.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points2_y, h_points2_y_col.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
#else
  // Allocate device memory
  uint32_t *d_points1_x, *d_points1_y, *d_points2_x, *d_points2_y;
  uint32_t *d_results_x, *d_results_y;

  cudaMalloc(&d_points1_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_points1_y, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_points2_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_points2_y, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_results_x, num_tests * limbs_per_field * sizeof(uint32_t));
  cudaMalloc(&d_results_y, num_tests * limbs_per_field * sizeof(uint32_t));

  // Copy row-major data to device
  cudaMemcpy(d_points1_x, h_points1_x.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points1_y, h_points1_y.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points2_x, h_points2_x.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points2_y, h_points2_y.data(), num_tests * limbs_per_field * sizeof(uint32_t), cudaMemcpyHostToDevice);
#endif

  // Launch kernel
  u32 threads_per_block = 256;
  u32 num_blocks = (num_tests + threads_per_block - 1) / threads_per_block;

  BatchPointAddition<<<num_blocks, threads_per_block>>>(
    d_points1_x, d_points1_y,
    d_points2_x, d_points2_y,
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
  cudaFree(d_points1_x);
  cudaFree(d_points1_y);
  cudaFree(d_points2_x);
  cudaFree(d_points2_y);
  cudaFree(d_results_x);
  cudaFree(d_results_y);

  ASSERT_EQ(num_correct, num_tests) << "Only " << num_correct << " out of " << num_tests << " tests passed";
}
