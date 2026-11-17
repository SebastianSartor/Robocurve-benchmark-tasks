"""Measure the overhead retrieval, staging, and distractor fractions."""

from dataclasses import dataclass

import numpy as np
import shapely

from cloth import panel_mesh
from layout import BIN_CENTER, BIN_HEIGHT, BIN_SIZE, WALL, ZONE, ZONE_SIZE


AREA_EPSILON = 1e-12
FRACTION_EPSILON = 1e-6
FACES = panel_mesh()[1]


@dataclass(frozen=True)
class Result:
    score: float
    target_retrieved: float
    target_staged: float
    distractor_max_outside: float
    panel_fractions_outside: tuple[float, ...]
    target_released: bool
    goal_met: bool


def footprint(vertices):
    """Project the surface to XY and count folded-over area once per panel."""
    triangles = np.asarray(vertices, dtype=float)[FACES, :2]
    if not np.isfinite(triangles).all():
        raise ValueError("Cloth nodes must be finite")
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    keep = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > AREA_EPSILON
    return shapely.union_all(shapely.polygons(triangles[keep]))


def box(center, size):
    half = np.asarray(size) / 2
    return shapely.box(*(center - half), *(center + half))


def score_footprints(shapes, target_slot, *, released):
    bin_box = box(BIN_CENTER, BIN_SIZE)
    staging = box(ZONE, ZONE_SIZE)

    def fraction(shape, region):
        return float(np.clip(shape.intersection(region).area / shape.area, 0, 1))

    target = shapes[target_slot]
    outside = tuple(1 - fraction(shape, bin_box) if shape.area > AREA_EPSILON else 1.0 for shape in shapes)
    r = outside[target_slot] if target.area > AREA_EPSILON else 0.0
    s = fraction(target, staging) if target.area > AREA_EPSILON else 0.0
    d = max(value for index, value in enumerate(outside) if index != target_slot)
    goal = min(r, s, 1 - d) >= 1 - FRACTION_EPSILON and released
    return Result((0.3 * r + 0.7 * s) * (1 - d), r, s, d, outside, released, goal)


def measure(layout, nodes, released):
    return score_footprints([footprint(panel) for panel in nodes], layout.target_slot,
                            released=released[layout.target_slot])


def verify_start(result, nodes, table_z):
    if max(result.panel_fractions_outside) > FRACTION_EPSILON or result.target_staged > FRACTION_EPSILON:
        raise RuntimeError(f"Mixed-bin start is invalid: {result}")
    for panel in nodes:
        heights = panel[:, 2] - table_z
        if heights.min() < WALL - 0.005 or heights.max() > BIN_HEIGHT:
            raise RuntimeError("A cloth panel did not settle in the open bin")
    covered = shapely.GeometryCollection()
    for panel in reversed(nodes):
        shape = footprint(panel)
        if shape.difference(covered).area <= AREA_EPSILON:
            raise RuntimeError("A cloth panel is completely hidden at reset")
        covered = covered.union(shape)
