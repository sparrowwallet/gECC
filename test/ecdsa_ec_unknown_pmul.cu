#include "gecc.h"
#include "gecc/support.h"
#include "gtest/gtest.h"
#include<cmath>

using namespace gecc;
using namespace gecc::arith;
using namespace gecc::ecdsa;

#include "ecdsa_test_constants.h"

template <typename ECDSA_EC_PMUL_Solver>
void test_ecdsa_ec_unknown_pmul() {
  u32 count = 1 << 22;

  ECDSA_EC_PMUL_Solver solver;
  ECDSA_EC_PMUL_Solver::initialize();
  
  // MAX_SM_NUMS=80: i=[12,17]+6
  // MAX_SM_NUMS=108: i=[12,17]+6
  // MAX_SM_NUMS=128: i=[11,16]+7
  for (int i = 11; i <= 16; i++) {
    count = MAX_SM_NUMS * (1<<i); //1<<18 ~ 1<<23
    printf("--------------------------- %u (%d << %d) --------------------------\n", count, MAX_SM_NUMS, i);

    solver.ec_pmul_random_init(RANDOM_S, RANDOM_KEY_X, RANDOM_KEY_Y, count);
    // warm up
    solver.ecdsa_ec_pmul(MAX_SM_NUMS<<2, 256, true);
    cudaDeviceSynchronize();

    double min_time = DBL_MAX;
    u32 min_blk_num = 0, min_thd_num = 0;
    for (u32 block_num = MAX_SM_NUMS; block_num <= MAX_SM_NUMS * 12; block_num += MAX_SM_NUMS) {
      for (u32 thread_num = 128; thread_num <= 256; thread_num *= 2) {
        if(block_num * thread_num > count) continue;
        if(32 * thread_num * 4 > 96*1024) continue;
        double min_elapsed = support::timeit(
          2, 2, [&]() {
            solver.ecdsa_ec_pmul(block_num, thread_num, true);
        });
        cudaDeviceSynchronize();
        if (min_time > min_elapsed) {
          min_time = min_elapsed;
          min_blk_num = block_num;
          min_thd_num = thread_num;
        }
        printf("unknown_point_mul: blc_num %u thd_num: %u time: %lf speed : %lf verifies/s\n", 
            block_num, thread_num, min_elapsed, (double)count / min_elapsed); 
      }
    }
    printf("unknown_point_mul: blc_num %u thd_num: %u time: %lf the fatested speed : %lf verifies/s\n", 
            min_blk_num, min_thd_num, min_time, (double)count / min_time);

    solver.ec_pmul_close();
  }
  
}

// Correctness test with debug output
template <typename ECDSA_EC_PMUL_Solver>
void test_ecdsa_ec_unknown_pmul_correctness() {
  u32 count = 3;  // Test with just 3 samples for easy verification

  ECDSA_EC_PMUL_Solver solver;
  ECDSA_EC_PMUL_Solver::initialize();

  printf("=== ECDSA EC Unknown Point Multiplication Correctness Test ===\n");
  printf("Testing %u point multiplications\n\n", count);

  // Initialize with test data
  solver.ec_pmul_random_init(RANDOM_S, RANDOM_KEY_X, RANDOM_KEY_Y, count);

  // Print inputs (first 3 samples)
  printf("Input scalars (s):\n");
  for (u32 i = 0; i < count; i++) {
    printf("  s[%u] = ", i);
    for (int j = 3; j >= 0; j--) {  // Print in big-endian order
      printf("%016llx", (unsigned long long)RANDOM_S[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("Input point X coordinates:\n");
  for (u32 i = 0; i < count; i++) {
    printf("  x[%u] = ", i);
    for (int j = 3; j >= 0; j--) {
      printf("%016llx", (unsigned long long)RANDOM_KEY_X[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("Input point Y coordinates:\n");
  for (u32 i = 0; i < count; i++) {
    printf("  y[%u] = ", i);
    for (int j = 3; j >= 0; j--) {
      printf("%016llx", (unsigned long long)RANDOM_KEY_Y[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  // Perform the computation
  solver.ecdsa_ec_pmul(MAX_SM_NUMS, 256, true);
  cudaDeviceSynchronize();

  // Get results back from device
  using Field = typename ECDSA_EC_PMUL_Solver::Field;
  const int field_limbs = Field::LIMBS_PER_LANE;

  uint32_t *h_result_x = new uint32_t[count * field_limbs];
  uint32_t *h_result_y = new uint32_t[count * field_limbs];

  // Results are stored in interleaved format in R0: X0, Y0, X1, Y1, X2, Y2, ...
  // solver.R0 is allocated with cudaMallocManaged, so we can access it directly from host
  for (u32 i = 0; i < count; i++) {
    memcpy(h_result_x + i * field_limbs, solver.R0 + i * field_limbs * 2, field_limbs * sizeof(uint32_t));
    memcpy(h_result_y + i * field_limbs, solver.R0 + i * field_limbs * 2 + field_limbs, field_limbs * sizeof(uint32_t));
  }

  printf("\nOutput result X coordinates (in Montgomery form):\n");
  for (u32 i = 0; i < count; i++) {
    printf("  result_x[%u] = ", i);
    for (int j = 7; j >= 0; j--) {  // u32 limbs, big-endian
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
  printf("To verify these results, update scripts/verify_gpu_output.py with the values above\n");
  printf("and run: cd scripts && python3 verify_gpu_output.py\n\n");

  delete[] h_result_x;
  delete[] h_result_y;
  solver.ec_pmul_close();
}

DEFINE_SECP256K1_FP(Fq_SECP256K1_1, FqSECP256K1, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::DEFAULT);
DEFINE_FP(Fq_SECP256K1_n, FqSECP256K1_n, u32, 32, LayoutT<1>, 8);
DEFINE_EC(G1_1, G1SECP256K1, Fq_SECP256K1_1, SECP256K1_CURVE, 1);
DEFINE_ECDSA(ECDSA_EC_PMUL_Solver, G1_1_G1SECP256K1, Fq_SECP256K1_1, Fq_SECP256K1_n);

TEST(ECDSA_EC_PMUL, Correctness) { test_ecdsa_ec_unknown_pmul_correctness<ECDSA_EC_PMUL_Solver>(); }
TEST(ECDSA_EC_PMUL, Performance) { test_ecdsa_ec_unknown_pmul<ECDSA_EC_PMUL_Solver>(); } 