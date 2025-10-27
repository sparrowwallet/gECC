# ECDSA EC Point Multiplication Correctness Testing

## Summary

Added a correctness test to `test/ecdsa_ec_unknown_pmul.cu` that prints inputs and outputs of EC point multiplication for Python verification.

## Bug Fixed

**Issue**: The original `generate_ecdsa_test()` function in `scripts/constants_generator.py` was generating random field elements for `RANDOM_KEY_X` and `RANDOM_KEY_Y`, not valid secp256k1 curve points.

**Fix**: Modified the function to use `ec.random_element()` to generate valid curve points:

```python
# OLD (WRONG):
random_key_x = [random.randint(0, f.p - 1)  for i in range(n)]
random_key_y = [random.randint(0, f.p - 1)  for i in range(n)]

# NEW (CORRECT):
random_key_x = []
random_key_y = []
for i in range(n):
    point = ec.random_element()
    random_key_x.append(point[0])
    random_key_y.append(point[1])
```

## Test Data

The test constants have been regenerated with valid curve points. You can verify this with:

```bash
cd scripts
python3 verify_new_test_points.py
```

## Running the Correctness Test

Build and run the test:

```bash
python dev-support/build.py -A <arch> -R -T ecdsa_ec_unknown_pmul_bk3_test -F ECDSA_EC_PMUL.Correctness
```

This will print:
- Input scalars (s) for 3 samples
- Input point X coordinates
- Input point Y coordinates
- Output result X coordinates (in Montgomery form)
- Output result Y coordinates (in Montgomery form)

## Python Verification

To verify the GPU results, you can now use the updated verification script:

```bash
cd scripts
python3 verify_ecdsa_pmul.py
```

Update the test data in `verify_ecdsa_pmul.py` with the values from your test run. The script will:
1. Convert GPU results from Montgomery form to normal form
2. Compute expected results using Python EC arithmetic
3. Compare and report any mismatches

## Files Modified

1. **scripts/constants_generator.py**
   - Fixed `generate_ecdsa_test()` to use valid curve points
   - Updated function signature to include `ec` parameter
   - Updated caller to pass `ec.G1_SECP256K1`

2. **test/ecdsa_test_constants.h**
   - Regenerated with valid secp256k1 curve points
   - All 3972 `RANDOM_KEY_X` and `RANDOM_KEY_Y` values are now valid curve points

3. **test/ecdsa_ec_unknown_pmul.cu**
   - Added `test_ecdsa_ec_unknown_pmul_correctness()` function
   - Prints inputs/outputs for first 3 samples
   - Accesses results from `solver.R0` member variable
   - Added `TEST(ECDSA_EC_PMUL, Correctness)` test case

## Files Created

1. **scripts/verify_ecdsa_pmul.py**
   - Python verification script for GPU results
   - Handles Montgomery form conversion
   - Uses secp256k1 curve arithmetic

2. **scripts/check_test_points.py**
   - Verifies if points are on the curve
   - Useful for debugging invalid curve point issues

3. **scripts/verify_new_test_points.py**
   - Verifies the regenerated test constants
   - Confirms all points are valid secp256k1 points

## Technical Notes

### Memory Layout

The `solver.R0` pointer contains affine coordinates in **interleaved format**:
- Point 0: X coordinates at `[0..field_limbs-1]`, Y coordinates at `[field_limbs..2*field_limbs-1]`
- Point 1: X coordinates at `[2*field_limbs..3*field_limbs-1]`, Y coordinates at `[3*field_limbs..4*field_limbs-1]`
- Point i: X at `[i*2*field_limbs..(i*2+1)*field_limbs-1]`, Y at `[(i*2+1)*field_limbs..(i+1)*2*field_limbs-1]`

For secp256k1 with `field_limbs=8` (u32):
- Point 0: X at [0..7], Y at [8..15]
- Point 1: X at [16..23], Y at [24..31]
- Point 2: X at [32..39], Y at [40..47]

The kernel uses `Affine::store(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx)` which stores coordinates in interleaved format.

### Montgomery Form

- GPU results are in Montgomery form
- Use `fq.from_mont(value)` in Python to convert to normal form
- Test input constants are in normal form
- GPU converts them to Montgomery form internally

### Memory Access

Since `solver.R0` is allocated with `cudaMallocManaged`, it can be accessed directly from the host after `cudaDeviceSynchronize()`. No explicit `cudaMemcpy` is required, though memcpy can be used to copy to separate host arrays.

### secp256k1 Curve Equation

```
y² = x³ + 7 (mod p)
where p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
```

For a point to be valid: `(y * y) % p == (x * x * x + 7) % p`
