"""
Modified fold file format specific to 22.5
Requirements:
 - clonable/hashable
 - no floats, only integers
"""
import matplotlib.pyplot as plt
from vertex4d import Vertex4D, AplusBsqrt2, SQRT2, R45, R225, split_triangle
import os
import copy
from fractions import Fraction
import numpy as np


class Fold225:
    def __init__(self, vertices, edges, faces=[], vertex_neighbors=[]):
        self.vertices = vertices
        self.edges = edges  # Set of tuples (v1, v2, line_type) representing edges between vertices
        self.faces = faces  # List of faces, where each face is a list of vertex indices
        self.vertex_neighbors = vertex_neighbors  # List of tuples (other_vertex_index, angle_in_22.5_degrees, line_type)

    def __repr__(self):
        return f"Fold225 with {len(self.vertices)} vertices and {len(self.edges)} edges"

    def __eq__(self, other):
        pass

    def __hash__(self):
        # TODO: improve this hash function to be invariant to rotation and mirror (so that folds that are "the same" hash the same)
        return hash((tuple(self.vertices), tuple(self.edges)))

    def render(self):
        """
        Convert similar to .cp file format: a list of edges where each edge is a list that contains the line type 'm','v','b','a' and the two vertices expressed as cartesian x1,y1,x2,y2 floats
        """
        rendered_edges = []
        for v1_idx, v2_idx, line_type in self.edges:
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]
            x1, y1 = v1.to_float()
            x2, y2 = v2.to_float()
            rendered_edges.append((line_type, x1, y1, x2, y2))
        return rendered_edges

    def get_vertex_neighbors(self):
        """
        Compute the neighbors for each vertex based on the edges.
        """
        neighbors = [[] for _ in self.vertices]
        for v1_idx, v2_idx, line_type in self.edges:
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]
            angle = v1.angle_to(v2)
            neighbors[v1_idx].append((v2_idx, angle, line_type))
            neighbors[v2_idx].append(
                (v1_idx, (angle + 8) % 16, line_type)
            )  # reverse direction
        self.vertex_neighbors = neighbors
        return neighbors

    def get_faces(self):
        """
        Compute faces based on the edges and vertex neighbors.
        """
        # This is a complex task and may require a more sophisticated algorithm.
        # For simplicity, we'll leave this unimplemented for now.
        pass


def plot_rendered_fold(fold: Fold225):
    """
    Plot a rendered fold using matplotlib
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    for edge in fold.render():
        line_type, x1, y1, x2, y2 = edge
        if line_type == "m":
            color = "red"
        elif line_type == "v":
            color = "blue"
        elif line_type == "b":
            color = "black"
        elif line_type == "a":
            color = "#c0c0c0"
        ax.plot([x1, x2], [y1, y2], color=color)

    for face in fold.faces:
        face_vertices = [fold.vertices[idx].to_float() for idx in face]
        xs, ys = zip(*face_vertices)
        ax.fill(xs, ys, color="green", alpha=0.2)

    for face in all_triangles.faces:
        vertices = [all_triangles.vertices[idx] for idx in face]
        new_edges = split_triangle(*vertices)

        for v1, v2, line_type in new_edges:
            x1, y1 = v1.to_float()
            x2, y2 = v2.to_float()
            plt.plot([x1, x2], [y1, y2], color="#c0c0c0", linestyle="--")
    ax.set_aspect("equal")
    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(renders_dir) if f.endswith(".png")]
    file_count = len(existing_files)
    filename = f"fold_render_{file_count}.png"
    filepath = os.path.join(renders_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved render to {filepath}")


if __name__ == "__main__":
    v1 = Vertex4D(0, 0, 0, 0)
    v2 = Vertex4D(1, 0, 0, 0)
    v3 = Vertex4D(1, 0, 1, 0)
    v4 = Vertex4D(0, 0, 1, 0)

    v5 = Vertex4D(1, 1, -1, 1)
    v6 = v2 * AplusBsqrt2(2, -1)
    v7 = Vertex4D(0, 1, 0, 0)
    all_triangles = Fold225(
        vertices=[v1, v2, v3, v4, v5, v6, v7],
        edges=[
            (0, 5, "b"),
            (5, 1, "b"),
            (1, 4, "b"),
            (4, 2, "b"),
            (2, 3, "b"),
            (3, 0, "b"),
            (0, 4, "m"),
            (0, 6, "m"),
            (3, 6, "m"),
            (4, 5, "m"),
            (6, 2, "m"),
            (6, 4, "m"),
        ],
        faces=[
            (0, 6, 3),  # expect 233
            (6, 2, 3),  # expect 135
            (6, 4, 2),  # expect 224
            (0, 4, 6),  # expect 134
            (0, 5, 4),  # expect 116
            # (5,1,4) # expect 224
        ],
    )

    # for face in all_triangles.faces:
    #     vertices = [all_triangles.vertices[idx] for idx in face]
    #     new_edges = split_triangle(*vertices)

    #     for v1, v2, line_type in new_edges:
    #         x1, y1 = v1.to_float()
    #         x2, y2 = v2.to_float()
    #         plt.plot([x1, x2], [y1, y2], color='orange', linestyle='--')

    # test_dict = {root_fold: "test"}
    # breakpoint()
    plot_rendered_fold(all_triangles)

    # ======= 
    # Below are the "derivations" for the new vertex locations of each of the 5 triangle types

    # ==== 224 triangle (isoceles right triangle) ====
    # A = Vertex4D(0,0,0,0)
    # B = Vertex4D(1,0,0,0)
    # C = Vertex4D(Fraction(1,2),0,Fraction(1,2),0)
    # v1 = Vertex4D(0,0,1,-1)
    # v2 = Vertex4D(1,-1,1,0)
    # v3 = Vertex4D(1,Fraction(-1,2),0,Fraction(1,2))
    # v4 = Vertex4D(Fraction(1,2),0,0,0)
    # v5 = Vertex4D(0,Fraction(1/2),0,Fraction(-1,2))
    # fold = Fold225(
    #     vertices = [A,B,C,v1,v2,v3,v4,v5],
    #     edges = [
    #         (0,1,'b'),(1,2,'b'),(2,0,'b'),
    #         (0,3,'a'),
    #         (1,4,'a'),
    #         (2,5,'a'),
    #         (2,6,'a'),
    #         (2,7,'a'),
    #     ],
    # )

    # ==== 233 triangle (isoceles 45 triangle) ====
    # A = Vertex4D(0,0,0,0)
    # B = Vertex4D(1,0,0,0)
    # C = Vertex4D(0,1,0,0)

    # # v1 = (B+C) * Fraction(1,2)  # midpoint of BC
    # # v3 = Vertex4D(Fraction(1,2),0,Fraction(1,2),0)
    # # v2 = v3 * AplusBsqrt2(2,-1)
    # # v4 = v2 * np.linalg.matrix_power(R45, 7)
    # # v5 = v3 * np.linalg.matrix_power(R45,7)
    # # print(v1,v2,v3,v4,v5)
    # v1 = Vertex4D(Fraction(1,2),Fraction(1,2),0,0)
    # v2 = Vertex4D(1,-1,1,0)
    # v3 = Vertex4D(Fraction(1,2),0,Fraction(1,2),0)
    # v4 = Vertex4D(-1,1,0,-1)
    # v5 = Vertex4D(0,Fraction(1,2),0,Fraction(-1,2))

    # fold = Fold225(
    #     vertices = [A,B,C,v1,v2,v3,v4,v5],
    #     edges = [
    #         (0,1,'b'),(1,2,'b'),(2,0,'b'),
    #         (0,3,'a'),
    #         (1,4,'a'),
    #         (1,5,'a'),
    #         (2,6,'a'),
    #         (2,7,'a'),
    #     ],
    # )

    # # ==== 134 triangle (22.5 right triangle) ====
    # A = Vertex4D(0,0,0,0)
    # B = Vertex4D(1,0,0,0)
    # C = Vertex4D(Fraction(1,2),Fraction(1,2),0,0)

    # # v1 = C * AplusBsqrt2(2,-1)
    # # v2 = v1 + (C-v1) * AplusBsqrt2(2,-1)
    # # v3 = B * Fraction(1,2)
    # # v4 = v3 * (AplusBsqrt2(0,1)/AplusBsqrt2(2,1) + 1)
    # # v5 = v4 + (B-v4) * Fraction(1,2)
    # # print(v1,v2,v3,v4,v5)
    # v1 = Vertex4D(Fraction(1,2),Fraction(1,2),Fraction(-1,2),Fraction(1,2))
    # v2 = Vertex4D(0,0,1,-1)
    # v3 = Vertex4D(Fraction(1,2),0,0,0)
    # v4 = Vertex4D(0,Fraction(1,2),0,Fraction(-1,2))
    # v5 = Vertex4D(Fraction(1,2),Fraction(1,4),0,Fraction(-1,4))

    # fold = Fold225(
    #     vertices = [A,B,C,v1,v2,v3,v4,v5],
    #     edges = [
    #         (0,1,'b'),(1,2,'b'),(2,0,'b'),
    #         (1,3,'a'),
    #         (1,4,'a'),
    #         (2,5,'a'),
    #         (2,6,'a'),
    #         (2,7,'a'),
    #     ],
    # )

    # # ==== 116 triangle (isoceles 135 triangle) ====
    # A = Vertex4D(0,0,0,0)
    # B = Vertex4D(1,0,0,0)
    # C = Vertex4D(Fraction(1,2),Fraction(1,2),Fraction(-1,2),Fraction(1,2))

    # # v3 = B * Fraction(1,2)
    # # v1 = v3 * AplusBsqrt2(2,-1)
    # # v2 = v1 + (v3-v1) * AplusBsqrt2(2,-1)
    # # v4 = v3 * 2 - v2
    # # v5 = v3 * 2 - v1
    # # print(v1,v2,v3,v4,v5)
    # v1 = Vertex4D(1,Fraction(-1,2),0,Fraction(1,2))
    # v2 = Vertex4D(-1,1,0,-1)
    # v3 = Vertex4D(Fraction(1,2),0,0,0)
    # v4 = Vertex4D(2,-1,0,1)
    # v5 = Vertex4D(0,Fraction(1,2),0,Fraction(-1,2))

    # fold = Fold225(
    #     vertices = [A,B,C,v1,v2,v3,v4,v5],
    #     edges = [
    #         (0,1,'b'),(1,2,'b'),(2,0,'b'),
    #         (2,3,'a'),
    #         (2,4,'a'),
    #         (2,5,'a'),
    #         (2,6,'a'),
    #         (2,7,'a'),
    #     ],
    # )

    # ==== 134 triangle (half kite) ====
    # A = Vertex4D(0,0,0,0)
    # B = Vertex4D(1,0,0,0)
    # C = Vertex4D(0,0,1,-1)

    # # v1 = Vertex4D(Fraction(1,2),Fraction(1,2),Fraction(-1,2),Fraction(1,2))
    # # v2 = B * (AplusBsqrt2(1,0)/AplusBsqrt2(1,1))
    # # v4 = (B-v2) * Fraction(1,2) + v2
    # # v3 = v2 + (v4-v2) * AplusBsqrt2(2,-1)
    # # v5 = v4 * 2 - v3
    # # print(v1,v2,v3,v4,v5)
    # v1 = Vertex4D(Fraction(1,2),Fraction(1,2),Fraction(-1,2),Fraction(1,2))
    # v2 = Vertex4D(-1,1,0,-1)
    # v3 = Vertex4D(2,-1,0,1)
    # v4 = Vertex4D(0,Fraction(1,2),0,Fraction(-1,2))
    # v5 = Vertex4D(-2,2,0,-2)

    # fold = Fold225(
    #     vertices = [A,B,C,v1,v2,v3,v4,v5],
    #     edges = [
    #         (0,1,'b'),(1,2,'b'),(2,0,'b'),
    #         (1,3,'a'),
    #         (2,4,'a'),
    #         (2,5,'a'),
    #         (2,6,'a'),
    #         (2,7,'a'),
    #     ],
    # )

    # plot_rendered_fold(fold)
