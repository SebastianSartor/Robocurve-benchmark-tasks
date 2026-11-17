"""Build the open bin, staging zone, and overlapping deformable cloth layers."""

import numpy as np

from cloth import ASSETS, FABRICS
from tape_tab import make_tab
from layout import BIN_CENTER, BIN_HEIGHT, BIN_SIZE, WALL, ZONE, ZONE_SIZE, TAPE_ANCHORS, layer_centers
from scene import MATERIALS, _bind_texture, make_station, panel_spawner


def staging_zone(table_z):
    from isaaclab.sim import CuboidCfg
    from isaaclab.sim.utils import clone, create_prim, get_current_stage
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from pxr import UsdGeom

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        stage = get_current_stage()
        root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
        plane = UsdGeom.Mesh.Define(stage, f"{prim_path}/Marking")
        x, y = (length / 2 for length in ZONE_SIZE)
        plane.CreatePointsAttr([(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)])
        plane.CreateFaceVertexCountsAttr([4])
        plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        plane.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        _bind_texture(stage, plane, ASSETS / "staging.png", np.array(((0, 0), (1, 0), (1, 1), (0, 1))))
        return root

    return Object(
        name="staging_zone",
        object_type=ObjectType.BASE,
        spawner_cfg=CuboidCfg(size=(*ZONE_SIZE, 0.001), func=spawn),
        initial_pose=Pose(position_xyz=(*ZONE, table_z + 0.0005)),
    )


def bin_parts():
    x, y = BIN_SIZE
    half_wall = WALL / 2
    mid_height = (BIN_HEIGHT + WALL) / 2
    side_height = BIN_HEIGHT - WALL
    return (
        ("Floor", (0, 0, half_wall), (x, y, WALL)),
        ("Near", (-(x / 2 - half_wall), 0, mid_height), (WALL, y, side_height)),
        ("Far", ((x / 2 - half_wall), 0, mid_height), (WALL, y, side_height)),
        ("Left", (0, (y / 2 - half_wall), mid_height), (x - 2 * WALL, WALL, side_height)),
        ("Right", (0, -(y / 2 - half_wall), mid_height), (x - 2 * WALL, WALL, side_height)),
    )


def mixed_bin(table_z):
    from isaaclab.sim import CuboidCfg
    from isaaclab.sim.utils import clone, create_prim, get_current_stage
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from pxr import Gf, UsdGeom, UsdPhysics

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        stage = get_current_stage()
        root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
        for label, center, size in bin_parts():
            cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{label}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([(0.29, 0.34, 0.39)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return root

    return Object(
        name="mixed_bin", object_type=ObjectType.BASE,
        spawner_cfg=CuboidCfg(size=(*BIN_SIZE, BIN_HEIGHT), func=spawn),
        initial_pose=Pose(position_xyz=(*BIN_CENTER, table_z)),
    )


def build_scene(task_name, layout, **kwargs):
    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.utils.pose import Pose
    from yam_arena.station import TABLE_TOP

    panels, tabs = [], []
    for index, (fabric_id, (x, y, z), anchor) in enumerate(
        zip(layout.slot_fabrics, layer_centers(), TAPE_ANCHORS, strict=True)
    ):
        fabric = FABRICS[fabric_id]
        center = (x, y, TABLE_TOP + z)
        tab = make_tab(index, center, anchor, MATERIALS[fabric.material][0])
        tabs.append(tab)
        panels.append(DeformableObject(
            name=f"panel_{index}", spawner_cfg=panel_spawner(fabric, tab_name=tab.name, anchor=anchor),
            initial_pose=Pose(position_xyz=center),
        ))
    return make_station(task_name, panels, [*tabs, mixed_bin(TABLE_TOP), staging_zone(TABLE_TOP)], **kwargs)
