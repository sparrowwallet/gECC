#!/usr/bin/env python3
"""
Verify the GPU output from the actual test run.
"""

import sys
sys.path.insert(0, '.')

import ec
import field
import constants

# secp256k1 curve
curve = ec.G1_SECP256K1
fq = field.Fq_SECP256K1
p = constants.SECP256K1_q
a = constants.SECP256K1_g1_a
b = constants.SECP256K1_g1_b

# GPU test output
test_data = [
    {
        's': 0x0278927476e92caa3912937a7f003e45c741ddc47d80d70ae8f35c0c7f3c78fd,
        'x': 0xef8ef523cd9e1a96dc497886b69cfc28474207c5679252541288869af65ee7f9,
        'y': 0xf59a57a32f25c0b0963dc44e5a268c1e258a118cfaecda3dadd2394b3e4bacc8,
        'result_x_mont': 0xbce9d493b5ebeeff5a5f128ce1405d9ee2df5318eaaa70a863ef06c805eb176a,
        'result_y_mont': 0x2baf4087630312bb8951f1d270442557da58cd4ee1a1523784a4e2b9eb0064e5,
    },
    {
        's': 0x1cec68f79cf23704c416d36bd4ac119e8d812385a3d1fb17695923e7be53095c,
        'x': 0xda73ee744524ed5ce8fa3c59c90596528988403a0e0bd063b2559bbf2a643d6f,
        'y': 0xe4fc93d3d9fd49dabae766763bf110305825fa07eb18eeefb90177c67f0d122a,
        'result_x_mont': 0xe098d7442ad7eb148aecbacdf94634065ae499e5349aa7c6b03a1794235f920a,
        'result_y_mont': 0x82cf6b14d196f6002a95dedbff7068b252cd09607957ae8aea362cd4dd4b5db7,
    },
    {
        's': 0x01d30c243eb1e07a216ba9c5d94cde6be89153e961e73e95e07f79e1c2c4fb3a,
        'x': 0x66a33482a42a3634d8a5ce069bd0c9af7c06a3c1df55d00f3f65d10a5d425faa,
        'y': 0x3ad4fc67c2cb6204638937595934bcc0545994b2d8e2aca5c36f292d71785bad,
        'result_x_mont': 0xf4b839fe9a983bbc799d72d4b3569a589ecd53c76d2b073867d273c738678c41,
        'result_y_mont': 0xb9a5c3c447721e070ae906d6edef15483e2ec4359e6992ad6207221ac2276100,
    },
]

def is_on_curve(x, y):
    """Check if point (x, y) is on the secp256k1 curve."""
    lhs = (y * y) % p
    rhs = (x * x * x + a * x + b) % p
    return lhs == rhs

print("="*70)
print("ECDSA EC Point Multiplication GPU Output Verification")
print("="*70)

# First, check if input points are valid
print("\nStep 1: Checking if input points are on the curve...")
all_valid_inputs = True
for idx, test in enumerate(test_data):
    x = test['x']
    y = test['y']
    if is_on_curve(x, y):
        print(f"  Point {idx}: ✓ ON CURVE")
    else:
        print(f"  Point {idx}: ✗ NOT ON CURVE")
        all_valid_inputs = False

if not all_valid_inputs:
    print("\n" + "="*70)
    print("ERROR: Input points are NOT valid secp256k1 curve points!")
    print("="*70)
    print("\nThe test is still using old invalid constants from ecdsa_test_constants.h")
    print("You need to regenerate this file properly with:")
    print("  cd scripts")
    print("  python3 constants_generator.py --out ../test")
    sys.exit(1)

print("\n✓ All input points are valid curve points!\n")

# Now verify the computation
print("Step 2: Verifying GPU computation against Python...")

all_passed = True
for idx, test in enumerate(test_data):
    print(f"\n--- Test {idx} ---")

    s = test['s']
    x = test['x']
    y = test['y']

    # Convert GPU result from Montgomery form
    result_x_normal = fq.from_mont(test['result_x_mont'])
    result_y_normal = fq.from_mont(test['result_y_mont'])

    print(f"Scalar s = {s:064x}")
    print(f"Point   x = {x:064x}")
    print(f"        y = {y:064x}")
    print(f"\nGPU Result (normal form):")
    print(f"  x = {result_x_normal:064x}")
    print(f"  y = {result_y_normal:064x}")

    # Compute expected with Python
    point_affine = (x, y)
    point_jac = curve.to_jacobian(point_affine)
    expected_jac = curve.multiply_jacobian(point_jac, s)
    expected_x, expected_y = curve.get_xy(expected_jac)

    print(f"\nPython Expected:")
    print(f"  x = {expected_x:064x}")
    print(f"  y = {expected_y:064x}")

    # Compare
    if result_x_normal == expected_x and result_y_normal == expected_y:
        print(f"\n✓ Test {idx} PASSED")
    else:
        print(f"\n✗ Test {idx} FAILED")
        all_passed = False
        if result_x_normal != expected_x:
            print(f"  X coordinate mismatch")
        if result_y_normal != expected_y:
            print(f"  Y coordinate mismatch")

print("\n" + "="*70)
if all_passed:
    print("✓ ALL TESTS PASSED - GPU implementation is correct!")
else:
    print("✗ SOME TESTS FAILED - there may be an issue with the GPU implementation")
print("="*70)

sys.exit(0 if all_passed else 1)
