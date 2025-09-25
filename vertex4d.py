"""
custom class for storing irrational 22.5 2d coordinates as rational 4d coordinates

rename to Math225
"""
import numpy as np
from fractions import Fraction

# projected such that (0,0,0,0) and (1,1,1,1) are opposite diagonal corners. Normal hypercube projection, rotated -22.5 degrees. This is a shortcut for readability, it doesn't scale correctly (cp won't end up in [0,1]x[0,1]) but it's just for display purposes anyways
# project4dto2d = np.array(
#     [
#         [np.cos(-np.pi / 8), -np.sin(-np.pi / 8)],
#         [np.sin(-np.pi / 8), np.cos(-np.pi / 8)],
#     ]
# ) @ np.array(
#     [[1, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2], [0, np.sqrt(2) / 2, 1, np.sqrt(2) / 2]]
# )
project4dto2d = np.array(
    [[1, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2], [0, np.sqrt(2) / 2, 1, np.sqrt(2) / 2]]
)
# Scale by sqrt(2)
SQRT2 = np.array([[0, 1, 0, -1], [1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 1, 0]])

# rotate 45 degrees counter clockwise
R45 = np.array([[0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
# rotate 22.5 degrees counter clockwise. Will change the scale
R225 = np.array([[1, 0, 0, -1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
# reflect across the x axis
REFLECTxAXIS = np.array([[1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0], [0, -1, 0, 0]])


class AplusBsqrt2:
    def __init__(self, A: Fraction, B: Fraction):
        self.A = A
        self.B = B

    def __repr__(self):
        return f"{self.A} + {self.B}√2"

    def __eq__(self, other):
        if isinstance(other, AplusBsqrt2):
            return (self.A, self.B) == (other.A, other.B)
        if isinstance(other, (int, Fraction)):
            return (self.A, self.B) == (other, 0)
        if isinstance(other, float):
            return (
                self.to_float() == other
            )  # may have precision issues. shouldn't be used
        raise ValueError(
            "Can only compare AplusBsqrt2 to AplusBsqrt2, int, float, or Fraction"
        )

    def __hash__(self):
        return hash((self.A, self.B))

    def __add__(self, other):
        if isinstance(other, AplusBsqrt2):
            return AplusBsqrt2(self.A + other.A, self.B + other.B)
        if isinstance(other, (int, Fraction)):
            return AplusBsqrt2(self.A + other, self.B)
        raise ValueError("Can only add AplusBsqrt2 to AplusBsqrt2, int, or Fraction")

    def __sub__(self, other):
        if isinstance(other, AplusBsqrt2):
            return AplusBsqrt2(self.A - other.A, self.B - other.B)
        if isinstance(other, (int, Fraction)):
            return AplusBsqrt2(self.A - other, self.B)
        raise ValueError(
            "Can only subtract AplusBsqrt2 by AplusBsqrt2, int, or Fraction"
        )

    def __mul__(self, other):
        if isinstance(other, AplusBsqrt2):
            return AplusBsqrt2(
                self.A * other.A + 2 * self.B * other.B,
                self.A * other.B + self.B * other.A,
            )
        if isinstance(other, (int, Fraction)):
            return AplusBsqrt2(self.A * other, self.B * other)
        raise ValueError(
            "Can only multiply AplusBsqrt2 by AplusBsqrt2, int, or Fraction"
        )

    def __truediv__(self, other):
        if isinstance(other, AplusBsqrt2):
            denom = other.A**2 - 2 * other.B**2
            if denom == 0:
                raise ZeroDivisionError("Cannot divide by zero AplusBsqrt2")
            newA = (self.A * other.A - 2 * self.B * other.B) * Fraction(1, denom)
            newB = (self.B * other.A - self.A * other.B) * Fraction(1, denom)
            return AplusBsqrt2(newA, newB)

        elif isinstance(other, (int, Fraction)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return AplusBsqrt2(self.A / other, self.B / other)

        else:
            raise ValueError(
                "Can only divide AplusBsqrt2 by AplusBsqrt2, int, or Fraction"
            )

    def __neg__(self):
        return AplusBsqrt2(-self.A, -self.B)

    def to_float(self):
        return float(self.A) + float(self.B) * np.sqrt(2)

    def sign(self):
        return np.sign(self.to_float())


# Precomputed trig function values for multiples of 22.5 degrees
# Note: technically 0 is not horizontal in the square, it's -22.5 because the square is rotated 22.5 degrees relative to the projected hypercube. But it's easier to think of 0 as aligned with the x component of the 4d projection

TAN_225 = {
    0: AplusBsqrt2(0, 0),  # tan( 0 *22.5)=0+0sqrt(2)
    1: AplusBsqrt2(-1, 1),  # tan( 1 *22.5)=-1+1sqrt(2)
    2: AplusBsqrt2(1, 0),  # tan( 2 *22.5)=1+0sqrt(2)
    3: AplusBsqrt2(1, 1),  # tan( 3 *22.5)=1
    # 4:                     vertical, tan is infinite
    5: AplusBsqrt2(-1, -1),  # tan( 5 *22.5)=-(sqrt(2)+1)
    6: AplusBsqrt2(-1, 0),  # -1
    7: AplusBsqrt2(1, -1),  # 1-sqrt(2)
}


class Vertex4D:
    def __init__(self, x, y, z, w):
        self.x = Fraction(x)
        self.y = Fraction(y)
        self.z = Fraction(z)
        self.w = Fraction(w)

        self.edges = []
        self.neighbors = []
        self.angles = []

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.w})"

    def __eq__(self, other):
        if not isinstance(other, Vertex4D):
            return NotImplemented
        return (self.x, self.y, self.z, self.w) == (other.x, other.y, other.z, other.w)

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.w))

    def __add__(self, other):
        if not isinstance(other, Vertex4D):
            raise ValueError("Can only add Vertex4D to Vertex4D")
        return Vertex4D(
            self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w
        )

    def __sub__(self, other):
        if not isinstance(other, Vertex4D):
            raise ValueError("Can only subtract Vertex4D from Vertex4D")
        return Vertex4D(
            self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w
        )

    def __mul__(self, other: int | Fraction | np.ndarray | AplusBsqrt2):
        if isinstance(other, (int, Fraction)):
            return Vertex4D(
                self.x * other, self.y * other, self.z * other, self.w * other
            )
        if isinstance(other, np.ndarray):
            if other.shape == (4, 4):
                x, y, z, w = other @ np.array([self.x, self.y, self.z, self.w])
                return Vertex4D(x, y, z, w)
            else:
                raise ValueError("Can only multiply Vertex4D by 4x4 numpy array")
        if isinstance(other, AplusBsqrt2):
            return self * other.A + (self * other.B) * SQRT2
        raise ValueError(
            "Can only multiply Vertex4D by int, Fraction, 4x4 numpy array, or AplusBsqrt2"
        )

    # TODO: implement division. For int, multiply by Fraction(1,other). For AplusBsqrt2, multiply by the conjugate and divide by the norm squared. For matrix, multiply by the inverse matrix if it exists.

    # TODO: play around with different possible definitions of multiplying two Vertex4D together. Dot product? Cross product? outer product? Quaternion multiplication?

    def angle_to(self, other):
        """
        Utilize 4d properties to compute the angle in units of 22.5 degrees from this vertex to another vertex in 4D space. If the angle is not an integer multiple of 22.5 degrees, return None
        """
        if not isinstance(other, Vertex4D):
            raise ValueError("Can only compute angle to another Vertex4D")
        if self == other:
            raise ValueError("Angle to self is undefined")
        diff = other - self

        X = AplusBsqrt2(diff.y - diff.w, diff.x)
        Y = AplusBsqrt2(diff.y + diff.w, diff.z)

        if X == 0:
            return 4 if Y.sign() > 0 else 12  # vertical up or down

        for k in (0, 1, 2, 3, 5, 6, 7):  # skip 4 (vertical)
            tan_k = TAN_225[k]
            # the slope of this kth direction is A_tan + B_tan*sqrt(2)
            # If the difference vector has the same slope, then the y component should equal this slope times the x component

            if Y == X * tan_k:
                return k if Y.sign() > 0 else (k + 8)
        return None
    def apply_transform(self, matrix: np.ndarray):
        """
        Apply a 4x4 transformation matrix to this vertex and return the resulting vertex
        """
        if not isinstance(matrix, np.ndarray) or matrix.shape != (4, 4):
            raise ValueError("Can only apply 4x4 numpy array as transformation matrix")
        x, y, z, w = matrix @ np.array([self.x, self.y, self.z, self.w])
        return Vertex4D(x, y, z, w)
    def to_float(self):
        """
        Convert to 2D cartesian coordinates as floats
        """
        result = project4dto2d @ np.array([self.x, self.y, self.z, self.w])
        return (float(result[0]), float(result[1]))
    def vec(self):
        """
        Convert to numpy array
        """
        return np.array([self.x, self.y, self.z, self.w]).T


# precomputed triangle types based on their angles (in 22.5 degree units)
# key is a tuple of the three angles sorted in increasing order of angle abc
# A is assumed to be at (0,0,0,0), B is at (1,0,0,0), and C is above the x axis
# value is a dict that contains a list of new vertices for each of the three corners A, B, and C to connect to
PRECOMPUTED_TRIANGLES = {
    # 224, isoceles right triangle
    (2, 2, 4): {
        "new_vertices_to_A": [Vertex4D(0, 0, 1, -1)],
        "new_vertices_to_B": [Vertex4D(1, -1, 1, 0)],
        "new_vertices_to_C": [
            Vertex4D(1, Fraction(-1, 2), 0, Fraction(1, 2)),
            Vertex4D(Fraction(1, 2), 0, 0, 0),
            Vertex4D(0, Fraction(1 / 2), 0, Fraction(-1, 2)),
        ],
    },
    # 233, isoceles 45 triangle
    (2, 3, 3): {
        "new_vertices_to_A": [Vertex4D(Fraction(1, 2), Fraction(1, 2), 0, 0)],
        "new_vertices_to_B": [
            Vertex4D(1, -1, 1, 0),
            Vertex4D(Fraction(1, 2), 0, Fraction(1, 2), 0),
        ],
        "new_vertices_to_C": [
            Vertex4D(-1, 1, 0, -1),
            Vertex4D(0, Fraction(1, 2), 0, Fraction(-1, 2)),
        ],
    },
    # 134, 22.5 right triangle
    (1, 3, 4): {
        "new_vertices_to_A": [],
        "new_vertices_to_B": [
            Vertex4D(Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2), Fraction(1, 2)),
            Vertex4D(0, 0, 1, -1),
        ],
        "new_vertices_to_C": [
            Vertex4D(Fraction(1, 2), 0, 0, 0),
            Vertex4D(0, Fraction(1, 2), 0, Fraction(-1, 2)),
            Vertex4D(Fraction(1, 2), Fraction(1, 4), 0, Fraction(-1, 4)),
        ],
    },
    # 116, 135 isoceles triangle
    (1, 1, 6): {
        "new_vertices_to_A": [],
        "new_vertices_to_B": [],
        "new_vertices_to_C": [
            Vertex4D(1, Fraction(-1, 2), 0, Fraction(1, 2)),
            Vertex4D(-1, 1, 0, -1),
            Vertex4D(Fraction(1, 2), 0, 0, 0),
            Vertex4D(2, -1, 0, 1),
            Vertex4D(0, Fraction(1, 2), 0, Fraction(-1, 2)),
        ],
    },
    # 125, half kite
    (1, 2, 5): {
        "new_vertices_to_A": [],
        "new_vertices_to_B": [
            Vertex4D(Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2), Fraction(1, 2))
        ],
        "new_vertices_to_C": [
            Vertex4D(-1, 1, 0, -1),
            Vertex4D(2, -1, 0, 1),
            Vertex4D(0, Fraction(1, 2), 0, Fraction(-1, 2)),
            Vertex4D(-2, 2, 0, -2),
        ],
    },
}


def transform_triangle(A: Vertex4D, B: Vertex4D, C: Vertex4D) -> np.ndarray:
    """
    Given a triangle ABC (sorted by increasing angle): construct the matrix that will transform it to the example triangle.

    A -> 0000, B -> 1000, C -> above x axis

    For full process of splitting the triangle:
    1. Identify which type of triangle it is (by angle) and also sort the vertices by angle to determine which is A, B, and C
    2. Compute the transformation matrix that takes ABC to the example triangle
    3. Take the new vertices provided by the example triangle and apply the inverse of the transformation matrix to get the new vertices in the original space. Also apply the translation ( += A)
    """
    angleAB = A.angle_to(B)
    angleAC = A.angle_to(C)
    if angleAB is None or angleAC is None:
        raise ValueError(
            "Cannot compute transformation matrix for triangle with non-22.5 degree angles"
        )

    # compute rotation component
    rotation_matrix = np.linalg.matrix_power(
        R45, 8 - angleAB // 2
    ) @ np.linalg.matrix_power(R225, angleAB % 2)

    # compute scaling component
    B_ = (B-A).apply_transform(rotation_matrix)

    scale_factor = AplusBsqrt2(B_.x,B_.y - B_.w)
    scaling_matrix = scale_factor.A * np.eye(4) + scale_factor.B * SQRT2

    matrix = rotation_matrix @ scaling_matrix

    # reflection component
    if not (angleAC - angleAB) % 16 < 8:
        matrix @= REFLECTxAXIS
    return matrix

def acute_diff(a1, a2, full=15):
    d = abs(a1 - a2)
    return min(d, full - d)

def split_triangle(v1: Vertex4D, v2: Vertex4D, v3: Vertex4D) -> list[list[Vertex4D]]:
    """
    Given a triangle ABC (not necessarily sorted by angle), return the 5 new edges
    """
    angle1 = acute_diff(v1.angle_to(v3), v1.angle_to(v2))
    angle2 = acute_diff(v2.angle_to(v1), v2.angle_to(v3))
    angle3 = acute_diff(v3.angle_to(v2), v3.angle_to(v1))

    angles_with_vertices = [
        (angle1, v1),
        (angle2, v2),
        (angle3, v3),
    ]
    angles_with_vertices.sort(key=lambda x: x[0])

    angles = tuple(a for a, _ in angles_with_vertices)
    A, B, C = (v for _, v in angles_with_vertices)
    new_vertices = PRECOMPUTED_TRIANGLES.get(angles)
    transformation = transform_triangle(A, B, C)
    # inverse = np.linalg.inv(transformation)

    new_edges = []
    for v, key in zip((A, B, C), ("new_vertices_to_A", "new_vertices_to_B", "new_vertices_to_C")):
        for new_vertex in new_vertices[key]:
            new_edges.append([v, new_vertex.apply_transform(transformation) + A,'a'])  # +A to undo translation. Transformation was computed with the assumption that A is at the origin
    return new_edges

if __name__ == "__main__":
    pass
