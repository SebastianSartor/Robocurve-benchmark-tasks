"""Track the cutout as the base moves and measure the patch from above."""

from dataclasses import dataclass

import numpy as np
import shapely

from layout import BASE_CENTER, BASE_THICKNESS, PATCH_START, PATCH_THICKNESS
from mesh import HOLE_LOOP, MARGIN_LOOP, task_meshes
from projection import AREA_EPSILON, footprint


FRACTION_EPSILON = 1e-6
BASE_FACES = task_meshes()[0][1]
PATCH_FACES = task_meshes()[1][1]


@dataclass(frozen=True)
class Result:
    score: float
    hole_coverage: float
    margin_square_coverage: float
    hole_center_xy_m: tuple[float, float]
    patch_released: bool
    resting_on_base: bool
    goal_met: bool


def ring(nodes, indices):
    return shapely.make_valid(shapely.Polygon(nodes[indices, :2]))


def coverage(patch, region):
    if region.area <= AREA_EPSILON:
        return 0.0
    return float(np.clip(patch.intersection(region).area / region.area, 0, 1))


def resting_on_base(base, patch):
    # Look for separated patch nodes near the top of the base at the same XY coordinates.
    triangles = base[BASE_FACES]
    origin = triangles[:, 0]
    a, b = triangles[:, 1] - origin, triangles[:, 2] - origin
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    nonvertical = abs(determinant) > AREA_EPSILON
    determinant = np.where(nonvertical, determinant, 1)
    delta = patch[:, None, :2] - origin[None, :, :2]
    u = (delta[:, :, 0] * b[:, 1] - delta[:, :, 1] * b[:, 0]) / determinant
    v = (a[:, 0] * delta[:, :, 1] - a[:, 1] * delta[:, :, 0]) / determinant
    inside = nonvertical & (u >= -1e-5) & (v >= -1e-5) & (u + v <= 1 + 1e-5)
    z = origin[:, 2] + u * a[:, 2] + v * b[:, 2]
    top = np.max(np.where(inside, z, -np.inf), axis=1)
    gap = patch[:, 2] - top
    expected = (BASE_THICKNESS + PATCH_THICKNESS) / 2
    support = patch[abs(gap - expected) <= 0.0025, :2]
    return bool(len(support) >= 3 and np.ptp(support, axis=0).min() >= 0.012)


def score_panels(base, patch, *, released):
    if not np.isfinite(base).all() or not np.isfinite(patch).all():
        raise ValueError("Cloth nodes must be finite")
    hole, required = ring(base, HOLE_LOOP), ring(base, MARGIN_LOOP)
    cover = footprint(patch, PATCH_FACES)
    h, m = coverage(cover, hole), coverage(cover, required)
    contact = resting_on_base(base, patch)
    full = min(h, m) >= 1 - FRACTION_EPSILON
    center = tuple(map(float, base[HOLE_LOOP, :2].mean(axis=0)))
    return Result((0.6 * h + 0.4 * m) if released else 0.0, h, m, center, released, contact,
                  bool(full and released and contact))


def measure(nodes, released):
    return score_panels(*nodes, released=released[1])


def verify_start(result, nodes, table_z):
    if result.hole_coverage > FRACTION_EPSILON or result.margin_square_coverage > FRACTION_EPSILON:
        raise RuntimeError(f"The patch overlaps the cutout at reset: {result}")
    for panel, center, (rest, _) in zip(nodes, (BASE_CENTER, PATCH_START), task_meshes()):
        heights = panel[:, 2] - table_z
        offset = panel[:, :2] - rest[:, :2] - center
        if heights.min() < -0.004 or heights.max() > 0.007 or np.linalg.norm(offset, axis=1).max() > 0.008:
            raise RuntimeError("A cloth panel did not settle at its starting position")
