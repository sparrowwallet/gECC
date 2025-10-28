# Fixed-Point Multiplication Analysis

## Summary

The fixed-point multiplication tests in `test/ecdsa_ec_fixed_pmul.cu` are producing incorrect results because the test infrastructure is not properly initialized for batch fixed-point multiplication.

## Root Cause

Fixed-point multiplication uses precomputed multiples of the secp256k1 generator point G:
- G, 2G, 4G, 8G, ..., 2^255·G

These precomputed values are stored in `test/ecdsa_constants.h` as `G1_1_G1SECP256K1.SIG_AFF[]` and loaded into device constant memory `ECDSACONST.d_mul_table[]` during `initialize()`.

**The problem**: The batch kernel `arith::fixedPMulByCombinedDAA` expects R1 to contain these precomputed multiples, but the current test calls `ec_pmul_random_init()` which overwrites R1 with input points instead.

## Two Fixed-Point Implementations

### 1. Device Function Approach (`fixed_point_mult`)
Used in: ECDSA signature signing kernels (`kernel_sig_sign`)

```cpp
__device__ static void fixed_point_mult(EC &r, Order &k, bool ec_operation) {
    for (u32 index = 0; index < Order::BITS; index++) {
        if (k.digits[index/32] & (1 << (index%32))) {
            q = get_d_mul_table(index);  // ← Reads from ECDSACONST.d_mul_table[]
            r = r + q;
        }
    }
}
```

**Works correctly** because it reads directly from device constant memory.

### 2. Batch Kernel Approach (`fixedPMulByCombinedDAA`)
Used in: Batch EC point multiplication tests

```cpp
__global__ void fixedPMulByCombinedDAA(typename EC::Base *R0,
                                        typename EC::Base *R1, ...) {
    for (int bit_index = 0; bit_index < Fr::BITS; bit_index++) {
        p2.x.load_arbitrary(R1, count, buc_index, lane_idx);  // ← Reads from R1
        // ... point addition ...
    }
}
```

**Broken** because R1 is not populated with precomputed values.

## Current Test Flow (Broken)

```cpp
// test/ecdsa_ec_fixed_pmul.cu
solver.ec_pmul_random_init(RANDOM_S, RANDOM_KEY_X, RANDOM_KEY_Y, count);
                           ↓
// include/gecc/ecdsa/gsv.h:ec_pmul_random_init()
processScalarPoint<<<>>>(..., R1, ...);  // ← Fills R1 with input points (WRONG!)
                           ↓
solver.ecdsa_ec_pmul(MAX_SM_NUMS, 256, false);  // false = fixed-point
                           ↓
// include/gecc/ecdsa/gsv.h:ecdsa_ec_pmul()
arith::fixedPMulByCombinedDAA<<<>>>(..., R1, ...);  // ← Expects R1 to have precomputed table!
```

## Solution Options

### Option 1: Create Proper Initialization Function
Add a new function `ec_fpmul_init()` that:
1. Allocates R0, R1, verify_t, etc.
2. Copies precomputed table from `ECDSACONST.d_mul_table[]` to R1
3. Copies scalar values to verify_t

```cpp
void ec_fpmul_init(const u64 s[][MAX_LIMBS], u32 count) {
    cudaMallocManaged(&verify_t, Order::SIZE * count);
    cudaMallocManaged(&R0, EC::Affine::SIZE * count);
    cudaMallocManaged(&R1, EC::Affine::SIZE * count);
    cudaMallocManaged(&acc_chain, EC::BaseField::SIZE * count * 2);
    cudaMallocManaged(&lambda_n, EC::BaseField::SIZE * count * 2);
    cudaMallocManaged(&lambda_den, EC::BaseField::SIZE * count * 2);

    // Copy scalars
    for (u32 i = 0; i < count; i++) {
        for (u32 j = 0; j < Order::LIMBS; j++) {
            verify_t[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(s[i])[j];
        }
    }

    // Copy precomputed table to R1
    // TODO: Need a kernel to copy ECDSACONST.d_mul_table[] → R1
    copy_precomputed_table_to_R1<<<...>>>(R1, count);
}
```

### Option 2: Modify Batch Kernel to Use Device Constant Memory
Change `fixedPMulByCombinedDAA` to read from `ECDSACONST.d_mul_table[]` instead of R1.

**Pros**: Simpler, avoids copying data
**Cons**: Changes kernel signature, may affect performance

### Option 3: Use Different Kernel for Testing
Don't use `fixedPMulByCombinedDAA` for testing. Instead, create a simple test kernel that uses `fixed_point_mult()` device function.

## Recommended Solution

**Option 3** is simplest for testing purposes. Create a new test kernel:

```cpp
template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_test_fixed_pmul(
    u32 count,
    typename Order::Base *scalars,
    typename EC::Base *results
) {
    u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance >= count) return;

    Order s;
    s.load_arbitrary(scalars, count, instance, 0);

    EC p = EC::zero();
    ECDSA_Solver::fixed_point_mult(p, s, true);  // ← Uses ECDSACONST.d_mul_table[]

    typename EC::Affine result = p.get_affine_x();
    result.store_arbitrary(results, count, instance, 0);
}
```

This kernel:
- Uses the existing `fixed_point_mult()` device function
- Reads directly from device constant memory (no R1 needed)
- Simple to test and verify

## Verification

The precomputed table in `test/ecdsa_constants.h` contains correct values:
- Entry 0: G (generator point)
- Entry 1: 2G
- Entry 2: 4G
- Entry i: 2^i · G

These can be verified against the secp256k1 standard.

## Files Involved

- `test/ecdsa_ec_fixed_pmul.cu` - Broken test file
- `test/ecdsa_constants.h` - Precomputed table (correct)
- `include/gecc/ecdsa/gsv.h` - Contains both `fixed_point_mult()` and `ecdsa_ec_pmul()`
- `include/gecc/arith/batch_ec.h` - Contains `fixedPMulByCombinedDAA` kernel
- `scripts/constants_generator.py` - Generates precomputed table (already working)

## Conclusion

The precomputed constants exist and are correct. The problem is purely in the test infrastructure not properly using them. The simplest fix is to create a dedicated test kernel that uses the working `fixed_point_mult()` device function instead of trying to fix the batch kernel initialization.
