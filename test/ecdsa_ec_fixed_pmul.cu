#include "gecc.h"
#include "gecc/support.h"
#include "gtest/gtest.h"
#include<cmath>

using namespace gecc;
using namespace gecc::arith;
using namespace gecc::ecdsa;

#include "ecdsa_test_constants.h"

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

DEFINE_SECP256K1_FP(Fq_SECP256K1_1, FqSECP256K1, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::DEFAULT);
DEFINE_FP(Fq_SECP256K1_n, FqSECP256K1_n, u32, 32, LayoutT<1>, 8);
DEFINE_EC(G1_1, G1SECP256K1, Fq_SECP256K1_1, SECP256K1_CURVE, 1);
DEFINE_ECDSA(ECDSA_EC_PMUL_Solver, G1_1_G1SECP256K1, Fq_SECP256K1_1, Fq_SECP256K1_n);
TEST(ECDSA_EC_PMUL, Performance) { test_ecdsa_ec_fixed_pmul<ECDSA_EC_PMUL_Solver>(); } 