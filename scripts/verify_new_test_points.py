#!/usr/bin/env python3
"""
Verify that the newly generated test points are valid secp256k1 points.
"""

import sys
sys.path.insert(0, '.')

import constants

# secp256k1 curve parameters
p = constants.SECP256K1_q  # Field prime
a = constants.SECP256K1_g1_a  # Curve parameter a (should be 0 for secp256k1)
b = constants.SECP256K1_g1_b  # Curve parameter b (should be 7 for secp256k1)

print("secp256k1 curve: y^2 = x^3 + ax + b")
print(f"  p = {hex(p)}")
print(f"  a = {a}")
print(f"  b = {b}")
print()

# Read first 3 test points from the generated file
import re

with open('../test/ecdsa_test_constants.h', 'r') as f:
    content = f.read()

# Extract RANDOM_KEY_X array
x_match = re.search(r'RANDOM_KEY_X\[3972\]\[MAX_LIMBS\] = \{\{([^}]+)\},\{([^}]+)\},\{([^}]+)\}', content)
# Extract RANDOM_KEY_Y array
y_match = re.search(r'RANDOM_KEY_Y\[3972\]\[MAX_LIMBS\] = \{\{([^}]+)\},\{([^}]+)\},\{([^}]+)\}', content)

if not x_match or not y_match:
    print("ERROR: Could not parse test constants file")
    sys.exit(1)

test_points = []
for i in range(1, 4):
    x_limbs = [int(v.strip()) for v in x_match.group(i).split(',')]
    y_limbs = [int(v.strip()) for v in y_match.group(i).split(',')]

    # Convert u64 limbs (little-endian) to big integer
    x = sum(limb << (64 * idx) for idx, limb in enumerate(x_limbs))
    y = sum(limb << (64 * idx) for idx, limb in enumerate(y_limbs))

    test_points.append({'x': x, 'y': y})

def is_on_curve(x, y):
    """Check if point (x, y) is on the secp256k1 curve."""
    lhs = (y * y) % p
    rhs = (x * x * x + a * x + b) % p
    return lhs == rhs

print("Checking if newly generated test points are valid secp256k1 curve points:\n")

all_valid = True
for idx, point in enumerate(test_points):
    x = point['x']
    y = point['y']

    print(f"Point {idx}:")
    print(f"  x = {x:064x}")
    print(f"  y = {y:064x}")

    if is_on_curve(x, y):
        print(f"  ✓ Point IS on the curve")
    else:
        print(f"  ✗ Point IS NOT on the curve")
        all_valid = False

        # Show what y^2 and x^3+ax+b actually are
        lhs = (y * y) % p
        rhs = (x * x * x + a * x + b) % p
        print(f"    y^2           = {lhs:064x}")
        print(f"    x^3 + ax + b  = {rhs:064x}")
    print()

print("="*70)
if all_valid:
    print("✓ All test points are valid secp256k1 curve points!")
    print("\nThe GPU verification should now work correctly.")
else:
    print("✗ Some test points are still invalid")
    sys.exit(1)
