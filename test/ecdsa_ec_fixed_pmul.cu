#include "gecc.h"
#include "gecc/support.h"
#include "gtest/gtest.h"
#include<cmath>

using namespace gecc;
using namespace gecc::arith;
using namespace gecc::ecdsa;

#include "ecdsa_fixed_test_constants.h"

// Define types first so they can be used in test functions
DEFINE_SECP256K1_FP(Fq_SECP256K1_1, FqSECP256K1, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::DEFAULT);
DEFINE_FP(Fq_SECP256K1_n, FqSECP256K1_n, u32, 32, LayoutT<1>, 8);
DEFINE_EC(G1_1, G1SECP256K1, Fq_SECP256K1_1, SECP256K1_CURVE, 1);
DEFINE_ECDSA(ECDSA_EC_PMUL_Solver, G1_1_G1SECP256K1, Fq_SECP256K1_1, Fq_SECP256K1_n);

template <typename ECDSA_EC_PMUL_Solver>
void test_ecdsa_ec_fixed_pmul() {
  u32 count = 1 << 22;

  ECDSA_EC_PMUL_Solver solver;
  ECDSA_EC_PMUL_Solver::initialize();

  // MAX_SM_NUMS=80: i=[12,17]+6
  // MAX_SM_NUMS=108: i=[12,17]+6
  // MAX_SM_NUMS=128: i=[11,16]+7
  for (int i = 11; i <= 16; i++) {
    count = MAX_SM_NUMS * (1<<i); //1<<18 ~ 1<<23
    printf("--------------------------- %u (%d << %d) --------------------------\n", count, MAX_SM_NUMS, i);

    // solver.verify_init(R, S, E, KEY_X, KEY_Y, count);
    solver.ec_pmul_random_init(RANDOM_S, RANDOM_KEY_X, RANDOM_KEY_Y, count);
    // warm up
    solver.ecdsa_ec_pmul(MAX_SM_NUMS<<2, 256, false);
    cudaDeviceSynchronize();

    double min_time = DBL_MAX;
    u32 min_blk_num = 0, min_thd_num = 0;
    for (u32 block_num = MAX_SM_NUMS; block_num <= MAX_SM_NUMS * 12; block_num += MAX_SM_NUMS) {
      for (u32 thread_num = 128; thread_num <= 256; thread_num *= 2) {
        if(block_num * thread_num > count) continue;
        double min_elapsed = support::timeit(
          2, 2, [&]() {
            solver.ecdsa_ec_pmul(block_num, thread_num, false);
        });
        cudaDeviceSynchronize();
        if (min_time > min_elapsed) {
          min_time = min_elapsed;
          min_blk_num = block_num;
          min_thd_num = thread_num;
        }
        printf("fixed_point_mul: blc_num %u thd_num: %u time: %lf speed : %lf verifies/s\n", 
            block_num, thread_num, min_elapsed, (double)count / min_elapsed); 
      }
    }
    printf("fixed_point_mul: blc_num %u thd_num: %u time: %lf the fatested speed : %lf verifies/s\n", 
            min_blk_num, min_thd_num, min_time, (double)count / min_time);

    solver.ec_pmul_close();
  }

}

// Test kernel that uses the fixed_point_mult device function
// This directly uses the precomputed table from ECDSACONST.d_mul_table[]
template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_test_fixed_pmul(
    u32 count,
    typename Order::Base *scalars,
    typename EC::Base *results
) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= count) return;

  Order s;
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    s.load_arbitrary(scalars, count, instance, 0);
  #else
    s.load(scalars + instance * Order::LIMBS, 0, 0, 0);
  #endif

  // Use the working fixed_point_mult device function
  // This reads from ECDSACONST.d_mul_table[] which contains precomputed multiples of G
  EC p = EC::zero();
  ECDSA_Solver::fixed_point_mult(p, s, true);

  // Convert to affine coordinates (computes both x and y)
  typename EC::Affine result = p.to_affine();
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    result.x.store_arbitrary(results, count, instance, 0);
    result.y.store_arbitrary(results + count * Field::LIMBS, count, instance, 0);
  #else
    result.x.store(results + instance * EC::Affine::LIMBS, 0, 0, 0);
    result.y.store(results + instance * EC::Affine::LIMBS + Field::LIMBS, 0, 0, 0);
  #endif
}

// Correctness test for fixed point multiplication
template <typename ECDSA_EC_PMUL_Solver>
void test_ecdsa_ec_fixed_pmul_correctness() {
  u32 count = 3;  // Test with just 3 samples for easy verification

  // Use the concrete types from the DEFINE_ECDSA macro
  using EC = G1_1_G1SECP256K1;
  using Field = typename ECDSA_EC_PMUL_Solver::Field;
  using Order = typename ECDSA_EC_PMUL_Solver::Order;

  ECDSA_EC_PMUL_Solver::initialize();

  printf("=== ECDSA EC Fixed Point Multiplication Correctness Test ===\n");
  printf("Testing %u point multiplications using generator G\n", count);
  printf("NOTE: Fixed-point multiplication ALWAYS computes s × G\n");
  printf("      (input points are ignored)\n\n");

  // Allocate memory for scalars and results
  typename Order::Base *d_scalars;
  typename EC::Base *d_results;

  cudaMallocManaged(&d_scalars, Order::SIZE * count);
  cudaMallocManaged(&d_results, EC::Affine::SIZE * count);

  // Copy test scalars to device
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    for (u32 j = 0; j < Order::LIMBS; j++) {
      for (u32 i = 0; i < count; i++) {
        d_scalars[j * count + i] = reinterpret_cast<const typename Order::Base *>(RANDOM_S[i])[j];
      }
    }
  #else
    for (u32 i = 0; i < count; i++) {
      for (u32 j = 0; j < Order::LIMBS; j++) {
        d_scalars[i * Order::LIMBS + j] = reinterpret_cast<const typename Order::Base *>(RANDOM_S[i])[j];
      }
    }
  #endif

  // Print input scalars
  printf("Input scalars (s):\n");
  for (u32 i = 0; i < count; i++) {
    printf("  s[%u] = ", i);
    for (int j = 3; j >= 0; j--) {
      printf("%016llx", (unsigned long long)RANDOM_S[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  // Launch kernel
  u32 block_num = (count + 255) / 256;
  u32 thread_num = 256;
  kernel_test_fixed_pmul<EC, Field, Order, ECDSA_EC_PMUL_Solver>
      <<<block_num, thread_num>>>(count, d_scalars, d_results);

  cudaDeviceSynchronize();
  if (cudaPeekAtLastError() != cudaSuccess) {
    printf("Kernel Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  // Read results
  const int field_limbs = Field::LIMBS_PER_LANE;
  uint32_t *h_result_x = new uint32_t[count * field_limbs];
  uint32_t *h_result_y = new uint32_t[count * field_limbs];

  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    for (u32 i = 0; i < count; i++) {
      for (u32 j = 0; j < field_limbs; j++) {
        h_result_x[i * field_limbs + j] = d_results[j * count + i];
        h_result_y[i * field_limbs + j] = d_results[count * field_limbs + j * count + i];
      }
    }
  #else
    for (u32 i = 0; i < count; i++) {
      memcpy(h_result_x + i * field_limbs, d_results + i * EC::Affine::LIMBS, field_limbs * sizeof(uint32_t));
      memcpy(h_result_y + i * field_limbs, d_results + i * EC::Affine::LIMBS + field_limbs, field_limbs * sizeof(uint32_t));
    }
  #endif

  printf("Output result X coordinates (in Montgomery form):\n");
  for (u32 i = 0; i < count; i++) {
    printf("  result_x[%u] = ", i);
    for (int j = 7; j >= 0; j--) {
      printf("%08x", h_result_x[i * field_limbs + j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("Output result Y coordinates (in Montgomery form):\n");
  for (u32 i = 0; i < count; i++) {
    printf("  result_y[%u] = ", i);
    for (int j = 7; j >= 0; j--) {
      printf("%08x", h_result_y[i * field_limbs + j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("=== Verification ===\n");
  printf("Run scripts/verify_fixed_point_correctness.py to verify these results.\n");
  printf("It will compute s × G for each scalar using Python and compare.\n\n");

  delete[] h_result_x;
  delete[] h_result_y;
  cudaFree(d_scalars);
  cudaFree(d_results);
}

TEST(ECDSA_EC_PMUL_FIXED, Correctness) { test_ecdsa_ec_fixed_pmul_correctness<ECDSA_EC_PMUL_Solver>(); }
TEST(ECDSA_EC_PMUL_FIXED, Performance) { test_ecdsa_ec_fixed_pmul<ECDSA_EC_PMUL_Solver>(); } 