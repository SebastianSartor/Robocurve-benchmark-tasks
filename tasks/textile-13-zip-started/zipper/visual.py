"""Render teeth and zipper tapes opening ahead of the measured slider position."""

import numpy as np

from .mechanism import LENGTH, PITCH, STEEL, TAPE


TOOTH_COUNT = round(LENGTH / PITCH)
CLOSED_HALF_GAP = 0.0025
OPEN_HALF_GAP = 0.017
TRANSITION = 0.025


def row_positions(slider, side):
    """Get one row of alternating teeth and its matching tape in rail coordinates."""
    phase = 0.25 if side > 0 else 0.75
    x = (np.arange(TOOTH_COUNT) + phase) * PITCH
    t = np.clip((x - slider) / TRANSITION, 0, 1)
    openness = t * t * (3 - 2 * t)
    y = side * (CLOSED_HALF_GAP + (OPEN_HALF_GAP - CLOSED_HALF_GAP) * openness)
    teeth = np.column_stack((x, y, np.full(TOOTH_COUNT, 0.022)))
    tape = np.column_stack((x, y + side * 0.008, np.full(TOOTH_COUNT, 0.019)))
    return np.vstack((teeth, tape)).astype(np.float32)


def author_row(stage, rail, side, slider):
    from pxr import Gf, UsdGeom, Vt

    path = rail.AppendChild("AnimatedZipper")
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    prototypes = []
    for name, size, color in (
        ("Tooth", (0.0034, 0.0068, 0.003), STEEL),
        ("Tape", (PITCH + 0.0005, 0.012, 0.002), TAPE),
    ):
        cube = UsdGeom.Cube.Define(stage, path.AppendPath(f"Prototypes/{name}"))
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        prototypes.append(cube.GetPath())
    instancer.CreatePrototypesRel().SetTargets(prototypes)
    instancer.CreateProtoIndicesAttr([0] * TOOTH_COUNT + [1] * TOOTH_COUNT)
    instancer.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(row_positions(slider, side)))


class ZipperVisual:
    def __init__(self, stage, zipper_path):
        from pxr import UsdGeom

        self.rows = [
            UsdGeom.PointInstancer.Get(stage, f"{zipper_path}/{rail}/AnimatedZipper").GetPositionsAttr()
            for rail in ("Base", "PinSide")
        ]
        self.last_slider = None

    def update(self, slider):
        from pxr import Vt

        slider = float(np.clip(slider, 0, LENGTH))
        if self.last_slider is not None and abs(slider - self.last_slider) < 0.0005:
            return
        for row, side in zip(self.rows, (1, -1)):
            row.Set(Vt.Vec3fArray.FromNumpy(row_positions(slider, side)))
        self.last_slider = slider
