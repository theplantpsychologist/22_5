import itertools

"""
Precompute all possible flat foldable 22.5 vertices. Ignore mountain/valley assignments.
There are 16 possible crease directions. 
Creases are flat foldable by Kawasaki if the alternating sum of absolute angles is 180 degrees.
To take advantage of the 22.5 degree system, let's represent angles as multiples of 22.5 degrees: 1 = 22.5 degrees, 2 = 45 degrees, ..., 8 = 180 degrees
"""
def alternating_sum(indices):
    # Sort indices to get the angles in order
    sorted_indices = sorted(indices)
    return sum(sorted_indices[1::2]) - sum(sorted_indices[0::2])

all_vertices = set()
for r in range(2, 17, 2):  # Only even numbers of creases can be flat-foldable
    for subset in itertools.combinations(range(16), r):
        if alternating_sum(subset) == 8:
            all_vertices.add(frozenset(subset))

print(all_vertices)
print(f"Found {len(all_vertices)} flat foldable vertices.")

# ==============

import json
import math


def abc2xyzw(ax, bx, cx, ay, by, cy):
    """
    Lift a vertex whose x and y coordinates are expressed in abc form to it's position (x,y,z,w) in 4d 
    """
    return (ax / cx, bx / cx + by / cy, ay / cy, -bx / cx + by / cy)


def abc2dec(a, b, c):
    """
    Convert values from abc form to decimal form
    """
    return (a + b * 2**0.5) / c


def generate_table():
    """
    Create a lookup table where the keys are decimal values and the values are the corresponding abc values
    """
    table = {}
    for a in range(-50, 50):
        for b in range(-50, 50):
            for c in range(1, 50):
                if math.gcd(a, b, c) > 1:
                    continue
                dec = abc2dec(a, b, c)
                if dec < 0 or dec > 1 or dec in table:
                    continue
                table[dec] = [a, b, c]
    sorted_table = dict(sorted(table.items()))
    with open("abc_table.json", "w") as file:
        json.dump(sorted_table, file)


generate_table()
