"""Fabric identities and the shared flat cloth rest mesh."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PANEL_SIZE = (0.140, 0.100)
MESH_SHAPE = (21, 15)
ASSETS = Path(__file__).resolve().parent / "assets"
CONTROL_HZ = 20
MAX_STEPS = 240 * CONTROL_HZ


@dataclass(frozen=True)
class Fabric:
    key: str
    material: str
    color: str
    pattern: str
    base: tuple[int, int, int]
    accent: tuple[int, int, int]


FABRICS = (
    Fabric("teal-cotton-stripes", "cotton", "teal", "white stripes", (20, 115, 116), (225, 240, 226)),
    Fabric("teal-cotton-rings", "cotton", "teal", "large white rings", (20, 115, 116), (225, 240, 226)),
    Fabric("rust-linen-plaid", "linen", "rust-orange", "dark plaid", (180, 82, 42), (80, 45, 41)),
    Fabric("purple-felt-dots", "felt", "purple", "pale polka dots", (114, 63, 148), (225, 195, 236)),
    Fabric("yellow-linen-plain", "linen", "golden-yellow", "a plain weave", (224, 180, 51), (240, 195, 63)),
)


def panel_mesh():
    nx, ny = MESH_SHAPE
    x, y = np.meshgrid(np.linspace(-PANEL_SIZE[0] / 2, PANEL_SIZE[0] / 2, nx),
                       np.linspace(-PANEL_SIZE[1] / 2, PANEL_SIZE[1] / 2, ny))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(nx * ny)))
    faces = []
    for row in range(ny - 1):
        for col in range(nx - 1):
            start = row * nx + col
            faces.extend(((start, start + 1, start + nx + 1), (start, start + nx + 1, start + nx)))
    return points, np.asarray(faces)
