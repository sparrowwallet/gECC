# !/bin/bash
export GTEST_ROOT=${HOME}/.local/opt/gtest
python3 ./dev-support/build.py -R -A 89

# -------------------------- ecdsa_sign_gen --------------------------------
echo "LOG: sig_gen breakdown analysis"
echo "LOG: BK1: rapid_ec:sig_gen + opt_modmul "
./build/Release/test/ecdsa_sign_bk1_test
echo "LOG: BK2: BK1 + batch_inv"
./build/Release/test/ecdsa_sign_bk2_test
echo "LOG: BK3: BK2 + batch_pmul optimization(kernel-fusion + mem-opt)"
./build/Release/test/ecdsa_sign_bk3_test

# # -------------------------- ecdsa_sign_verify --------------------------------
echo "LOG: sig_verify breakdown analysis"
echo "LOG: BK1: sig_verify: rapid_ec:sig_verify + opt_modmul"
./build/Release/test/ecdsa_verify_bk1_test
echo "LOG: BK2: BK1 + batch fixed-point mul (no mem opt)"
./build/Release/test/ecdsa_verify_bk2_test
echo "LOG: BK3: BK2 + batch_pmul optimization(kernel-fusion + mem-opt)"
./build/Release/test/ecdsa_verify_bk3_test

# -------------------------- ecdsa_unknown_pmul --------------------------------
echo "LOG: EC UPMUL breakdown analysis"
echo "LOG: BK1: pmul_naf + opt_modmul"
./build/Release/test/ecdsa_ec_unknown_pmul_bk1_test
echo "LOG: BK2: batch-upmul: no optimization"
./build/Release/test/ecdsa_ec_unknown_pmul_bk2_test
echo "LOG: BK3: batch-upmul: kernel-fusion optimization"
./build/Release/test/ecdsa_ec_unknown_pmul_bk3_test
echo "LOG: BK4: batch-upmul: kernel-fusion optimization + memory management"
./build/Release/test/ecdsa_ec_unknown_pmul_bk4_test

# -------------------------- ecdsa_fixed_pmul --------------------------------
echo "LOG: EC FPMUL breakdown analysis"
echo "LOG: BK1: fpmul + opt_modmul"
./build/Release/test/ecdsa_ec_fixed_pmul_bk1_test
echo "LOG: BK2: batch-fpmul: no optimization"
./build/Release/test/ecdsa_ec_fixed_pmul_bk2_test
echo "LOG: BK3: batch-fpmul: kernel-fusion optimization"
./build/Release/test/ecdsa_ec_fixed_pmul_bk3_test
echo "LOG: BK4: batch-fpmul: kernel-fusion optimization + memory management"
./build/Release/test/ecdsa_ec_fixed_pmul_bk4_test

# -------------------------- fp_test --------------------------------
echo "LOG: Fp analysis"
./build/Release/test/fp_test

## Profiling
# ./build/Release/test/modinv_data_parallel_profiling_test
