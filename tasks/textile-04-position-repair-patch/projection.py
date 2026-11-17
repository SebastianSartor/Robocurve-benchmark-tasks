"""Orthographic cloth projection with folded layers counted once."""

import numpy as np
import shapely


AREA_EPSILON = 1e-12


def footprint(vertices, faces):
    """Project the surface to XY and count folded-over area once per panel."""
    triangles = np.asarray(vertices, dtype=float)[faces, :2]
    if not np.isfinite(triangles).all():
        raise ValueError("Cloth nodes must be finite")
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    keep = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > AREA_EPSILON
    return shapely.union_all(shapely.polygons(triangles[keep]))


def box(center, size):
    half = np.asarray(size) / 2
    return shapely.box(*(center - half), *(center + half))
