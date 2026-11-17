"""Put the cutout and loose repair patch on the YAM table."""

from layout import BASE_CENTER, BASE_SIZE, BASE_THICKNESS, PATCH_SIZE, PATCH_START, PATCH_THICKNESS
from layout import PATCH_TAB_ANCHOR, PATCH_TAB_MASS, PATCH_TAB_SIZE
from scene import cloth_spawner, make_station
from tape_tab import make_tab


def build_scene(task_name, meshes, **kwargs):
    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.utils.pose import Pose
    from yam_arena.station import TABLE_TOP

    placements = (
        (BASE_CENTER, BASE_SIZE, BASE_THICKNESS, "base-cotton.png", 9.54e4, 195.3),
        (PATCH_START, PATCH_SIZE, PATCH_THICKNESS, "green-patch.png", 1.43e5, 187.5),
    )
    spawn_height = TABLE_TOP + 0.008
    tab = make_tab(
        1, (*PATCH_START, spawn_height), PATCH_TAB_ANCHOR, PATCH_THICKNESS,
        size=PATCH_TAB_SIZE, mass=PATCH_TAB_MASS,
    )
    panels = [
        DeformableObject(
            name=f"panel_{i}",
            spawner_cfg=cloth_spawner(
                *mesh, size=size, texture=texture, thickness=thickness, bend=bend, density=density,
                tab_name=tab.name if i == 1 else None,
            ),
            initial_pose=Pose(position_xyz=(*center, spawn_height)),
        )
        for i, (mesh, (center, size, thickness, texture, bend, density)) in enumerate(zip(meshes, placements))
    ]
    return make_station(task_name, panels, [tab], **kwargs)
