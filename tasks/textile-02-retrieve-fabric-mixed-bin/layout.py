"""Seeded start positions and work orders for the mixed-bin task."""

from dataclasses import dataclass

import numpy as np

from cloth import FABRICS
from tape_tab import TAB_SIZE


BIN_CENTER = (0.280, -0.120)
BIN_SIZE = (0.240, 0.180)
BIN_HEIGHT = 0.080
WALL = 0.004
ZONE = (0.280, 0.150)
ZONE_SIZE = (0.180, 0.140)
# Offsets from bin center in bottom-to-top order, leaving part of each layer visible.
LAYER_OFFSETS = ((-0.043, -0.032), (0.043, 0.032), (-0.043, 0.032),
                 (0.043, -0.032), (0.006, 0.003))
LAYER_PITCH = 0.009
# Offsets in each panel's frame. The upper panels leave these areas exposed.
TAPE_ANCHORS = ((-0.039, -0.024), (0.054, 0.016), (-0.044, 0.023),
                (0.054, -0.020), (0.002, 0.008))


@dataclass(frozen=True)
class Layout:
    seed: int
    slot_fabrics: tuple[int, ...]
    target_fabric: int

    @property
    def target_slot(self):
        return self.slot_fabrics.index(self.target_fabric)

    @property
    def instruction(self):
        target = FABRICS[self.target_fabric]
        return (
            f"Retrieve the {target.color} {target.material} panel with {target.pattern} "
            "from the gray bin. Grip the raised tape tab on the matching panel and lift it clear of the bin. "
            "Move other panels aside if needed. Place the entire target panel inside "
            "the green STAGING zone and let go. Leave every other panel in the bin. "
            "If you take a different panel out, put it back."
        )

    def describe(self):
        return {
            "seed": self.seed,
            "instruction": self.instruction,
            "bin_center_m": BIN_CENTER,
            "bin_outer_size_m": (*BIN_SIZE, BIN_HEIGHT),
            "staging_center_m": ZONE,
            "staging_size_m": ZONE_SIZE,
            "target_slot": self.target_slot,
            "tape_tab_size_m": TAB_SIZE,
            "panels": [
                {"fabric": FABRICS[fabric].key, "center_above_table_m": center, "tape_anchor_xy_m": anchor}
                for fabric, center, anchor in zip(self.slot_fabrics, layer_centers(), TAPE_ANCHORS, strict=True)
            ],
        }


def make_layout(seed):
    rng = np.random.default_rng(seed)
    return Layout(seed, tuple(map(int, rng.permutation(len(FABRICS)))), int(rng.integers(len(FABRICS))))


def layer_centers():
    return tuple((BIN_CENTER[0] + x, BIN_CENTER[1] + y, WALL + 0.008 + i * LAYER_PITCH)
                 for i, (x, y) in enumerate(LAYER_OFFSETS))
