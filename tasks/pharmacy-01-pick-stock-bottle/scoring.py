"""Measure final bottle placement from the projected bodies and bases."""

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import MultiPoint, box

from layout import BODY_HEIGHT, BODY_RADIUS, BOTTLE_HEIGHT, CAP_HEIGHT, CAP_RADIUS, SLOTS, SLOT_SIZE, ZONE_CENTER, ZONE_SIZE


UPRIGHT_DEG = 15.0
SUPPORT_TOLERANCE = 0.006
REST_SPEED = 0.010
REST_SPIN = 0.5
RELEASE_CLEARANCE = 0.030
ANGLES = np.linspace(0, 2 * math.pi, 96, endpoint=False)


def ring(radius, z):
    return np.column_stack((radius * np.cos(ANGLES), radius * np.sin(ANGLES), np.full(ANGLES.size, z)))


BOTTOM = -BOTTLE_HEIGHT / 2
BASE_RING = ring(BODY_RADIUS, BOTTOM)
BODY_RINGS = np.vstack((BASE_RING, ring(BODY_RADIUS, BOTTOM + BODY_HEIGHT),
                        ring(CAP_RADIUS, -BOTTOM - CAP_HEIGHT), ring(CAP_RADIUS, -BOTTOM)))
RING_COUNT = len(BASE_RING)


@dataclass(frozen=True)
class Reading:
    score: float
    goal_met: bool
    stage_score: float
    distractor_penalty: float
    displaced_distractors: tuple[int, ...]
    target_in_zone: bool
    target_out_of_slot: bool
    target_upright: bool
    target_released: bool
    target_resting_on_table: bool


def rotate(points, quat):
    axis, scalar = quat[:3], quat[3]
    cross = np.cross(axis, points)
    return points + 2 * (scalar * cross + np.cross(axis, cross))


def rectangle(center, size):
    half = np.asarray(size) / 2
    center = np.asarray(center)
    return box(*(center - half), *(center + half))


def read_score(layout, poses, velocities, hands, table_z):
    rings = [rotate(BODY_RINGS, pose[3:]) + pose[:3] for pose in poses]
    out = tuple(
        not rectangle(xy, SLOT_SIZE).intersects(MultiPoint(points[:RING_COUNT, :2]).convex_hull)
        for xy, points in zip(SLOTS, rings, strict=True)
    )
    target_index = layout.target_slot
    target = rings[target_index]
    velocity = velocities[target_index]
    low, high = target.min(axis=0), target.max(axis=0)
    hand_gaps = np.maximum(np.maximum(low - hands, hands - high), 0.0)
    released = bool(np.min(np.linalg.norm(hand_gaps, axis=1)) > RELEASE_CLEARANCE)
    resting = bool(abs(low[2] - table_z) <= SUPPORT_TOLERANCE
                   and np.linalg.norm(velocity[:3]) <= REST_SPEED and np.linalg.norm(velocity[3:]) <= REST_SPIN)
    axis_z = rotate(np.asarray([0.0, 0.0, 1.0]), poses[target_index, 3:])[2]
    upright = bool(axis_z >= math.cos(math.radians(UPRIGHT_DEG)))
    in_zone = bool(rectangle(ZONE_CENTER, ZONE_SIZE).covers(box(*low[:2], *high[:2])))
    staged = in_zone and resting and released
    if staged:
        stage_score = 1.0 if upright else 0.5
    elif out[target_index] and not in_zone:
        stage_score = 0.25
    else:
        stage_score = 0.0
    wrong = tuple(i for i, left_slot in enumerate(out) if i != target_index and left_slot)
    penalty = min(1.0, len(wrong) / 2)
    score = stage_score * (1 - penalty)
    return Reading(score, score == 1.0, stage_score, penalty, wrong, in_zone, out[target_index], upright,
                   released, resting)
