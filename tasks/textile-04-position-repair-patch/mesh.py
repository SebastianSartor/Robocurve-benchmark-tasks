"""Build the cloth geometry and identify the material points around the cutout."""

from functools import lru_cache

import numpy as np

from layout import BASE_SIZE, HOLE_SIZE, MARGIN, PATCH_SIZE


def grid_mesh(x, y, *, hole_half=0):
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    faces = []
    for row in range(len(y) - 1):
        for col in range(len(x) - 1):
            if abs((x[col] + x[col + 1]) / 2) < hole_half and abs((y[row] + y[row + 1]) / 2) < hole_half:
                continue
            i = row * len(x) + col
            j = i + len(x)
            faces.extend(((i, i + 1, j + 1), (i, j + 1, j)))
    used, remapped = np.unique(faces, return_inverse=True)
    return points[used], remapped.reshape(-1, 3)


def repair_axis(length, divisions):
    half = HOLE_SIZE / 2
    rim = half + MARGIN
    # Keep exact coordinates for both squares and omit generic grid lines too close to them.
    ticks = np.linspace(-length / 2, length / 2, divisions)
    limits = np.array((-rim, -half, half, rim))
    keep = (np.abs(ticks[:, None] - limits).min(axis=1) >= MARGIN).nonzero()[0]
    return np.sort(np.r_[ticks[keep], limits])


@lru_cache(maxsize=1)
def task_meshes():
    base = grid_mesh(repair_axis(BASE_SIZE[0], 21), repair_axis(BASE_SIZE[1], 15), hole_half=HOLE_SIZE / 2)
    patch = grid_mesh(*(np.linspace(-side / 2, side / 2, 11) for side in PATCH_SIZE))
    return base, patch


def rim_loop(half):
    xy = task_meshes()[0][0][:, :2]
    x, y = xy.T
    selected = np.flatnonzero(np.isclose(np.maximum(abs(x), abs(y)), half))
    angles = np.arctan2(y[selected], x[selected])
    return selected[np.argsort(angles)]


HOLE_LOOP = rim_loop(HOLE_SIZE / 2)
MARGIN_LOOP = rim_loop(HOLE_SIZE / 2 + MARGIN)
