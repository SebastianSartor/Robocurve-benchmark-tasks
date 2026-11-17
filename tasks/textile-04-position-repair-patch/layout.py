"""Fixed repair-patch layout in YAM station coordinates."""

from dataclasses import dataclass
from pathlib import Path


ASSETS = Path(__file__).resolve().parent / "assets"
CONTROL_HZ = 20
MAX_STEPS = 180 * CONTROL_HZ
BASE_SIZE = (0.140, 0.100)
BASE_CENTER = (0.170, 0.000)
BASE_THICKNESS = 0.0032
PATCH_SIZE = (0.035, 0.035)
PATCH_START = (0.170, 0.091)
PATCH_THICKNESS = 0.0024
PATCH_TAB_SIZE = (0.013, 0.004, 0.015)
PATCH_TAB_BASE = (0.013, 0.014)
PATCH_TAB_ANCHOR = (-0.008, 0.008)
PATCH_TAB_MASS = 0.00085
HOLE_SIZE = 0.025
MARGIN = 0.002
INSTRUCTION = (
    "Grip the raised tape tab on the loose green square patch and place the patch over the square hole in the blue cloth. "
    "Cover the entire opening, with at least 2 millimeters of patch extending past each side. "
    "Smooth the patch flat against the blue cloth and let go."
)


@dataclass(frozen=True)
class Layout:
    seed: int
    instruction: str = INSTRUCTION

    def describe(self):
        return {
            "seed": self.seed, "instruction": self.instruction,
            "base_center_m": BASE_CENTER, "base_size_m": BASE_SIZE,
            "hole_center_in_base_m": (0, 0), "hole_size_m": HOLE_SIZE,
            "margin_m": MARGIN, "patch_start_m": PATCH_START, "patch_size_m": PATCH_SIZE,
            "patch_tab_xy_offset_m": PATCH_TAB_ANCHOR, "patch_tab_size_m": PATCH_TAB_SIZE,
        }


def make_layout(seed):
    return Layout(seed)
