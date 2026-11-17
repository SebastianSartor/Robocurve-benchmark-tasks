"""Seeded shelf and simulated prescription for PharmacyBench Task 01."""

from dataclasses import dataclass
from itertools import product
import random


CONTROL_HZ = 20
MAX_STEPS = 120 * CONTROL_HZ
INSTRUCTION = (
    "Read the name and strength on the prescription card. Pick the stock bottle that matches both. "
    "Place it upright, completely inside the green STAGING zone, and let go. "
    "Leave every other bottle in its marked shelf slot."
)
SLOTS = tuple((0.300, (i - 2.5) * 0.070) for i in range(6))
SLOT_SIZE = (0.056, 0.056)
SHELF_CENTER = (0.300, 0.0)
SHELF_SIZE = (0.100, 0.430, 0.012)
ZONE_CENTER = (0.155, 0.0)
ZONE_SIZE = (0.120, 0.120)
CARD_CENTER = (0.155, 0.186)
CARD_SIZE = (0.097, 0.124)  # table x, y; portrait height and printed width
BODY_RADIUS = 0.023
CAP_RADIUS = 0.022
BOTTLE_HEIGHT = 0.086
BODY_HEIGHT = 0.073
CAP_HEIGHT = 0.014
BOTTLE_MASS = 0.085


@dataclass(frozen=True)
class Stock:
    name: str
    strength_mg: int

    @property
    def key(self):
        return f"{self.name.lower()}-{self.strength_mg}"

    @property
    def strength(self):
        return f"{self.strength_mg} mg"


MATCH_SET = tuple(Stock(name, strength) for name, strength in product(("NERAVEX", "NERAVIX"), (10, 25)))
STOCK = (*MATCH_SET, Stock("LUMETRA", 10), Stock("SELDRIN", 25))


@dataclass(frozen=True)
class Layout:
    seed: int
    target: Stock
    bottles: tuple[Stock, ...]

    @property
    def target_slot(self):
        return self.bottles.index(self.target)

    def describe(self):
        return {
            "seed": self.seed,
            "instruction": INSTRUCTION,
            "prescription": f"{self.target.name} {self.target.strength}",
            "target_slot": self.target_slot,
            "shelf_center_m": SHELF_CENTER,
            "shelf_size_m": SHELF_SIZE,
            "slot_size_m": SLOT_SIZE,
            "bottles": [
                {"slot": i, "name": stock.name, "strength": stock.strength, "xy_m": xy}
                for i, (stock, xy) in enumerate(zip(self.bottles, SLOTS, strict=True))
            ],
            "staging_center_m": ZONE_CENTER,
            "staging_size_m": ZONE_SIZE,
            "card_center_m": CARD_CENTER,
            "card_size_m": CARD_SIZE,
        }


def make_layout(seed):
    rng = random.Random(seed)
    target = rng.choice(MATCH_SET)
    bottles = tuple(rng.sample(STOCK, k=len(STOCK)))
    return Layout(seed, target, bottles)
