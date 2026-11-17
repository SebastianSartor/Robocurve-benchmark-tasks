"""Measure each panel's projected fabric area in its assigned zone."""

import numpy as np
import shapely


def projected_fraction(vertices, faces, center, size):
    """Return the fraction inside the zone, excluding holes and counting overlaps once."""
    triangles = np.asarray(vertices, dtype=float)[faces, :2]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    area2 = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    triangles = triangles[np.abs(area2) > 1e-12]
    if not len(triangles):
        return 0.0
    fabric = shapely.union_all(shapely.polygons(triangles))
    lo = np.asarray(center) - np.asarray(size) / 2
    hi = np.asarray(center) + np.asarray(size) / 2
    inside = fabric.intersection(shapely.box(*lo, *hi)).area
    return float(np.clip(inside / fabric.area, 0.0, 1.0))


def sort_result(nodes, faces, hands, damaged_slot, zones, zone_size, *, tabs):
    panels = {}
    for label, slot in (("Damaged", damaged_slot), ("Undamaged", 1 - damaged_slot)):
        vertices = nodes[slot]
        grasp_region = np.vstack((vertices, tabs[slot]))
        clearance = float(np.linalg.norm(grasp_region[:, None, :] - hands, axis=2).min())
        panels[label] = {
            "fraction_in_zone": projected_fraction(vertices, faces[slot], zones[label][0], zone_size),
            "released": clearance > 0.08,
            "tcp_clearance_m": clearance,
        }
    score = 0.5 * sum(panel["fraction_in_zone"] for panel in panels.values() if panel["released"])
    return {"score": score, "panels": panels}
