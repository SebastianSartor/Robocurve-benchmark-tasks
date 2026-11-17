"""Raised, folded cloth-tape tabs for the TextileBench pilots."""

import numpy as np

# 25 mm wide in X, 4 mm thick in Y, and 18 mm tall.
TAB_SIZE = (0.025, 0.004, 0.018)
TAB_COLOR = (0.98, 0.68, 0.18)
TAB_MASS = 0.001
BASE_PATCH = (0.025, 0.019)


def tab_offset(anchor, cloth_thickness):
    """Position the grip block just above the cloth mid-surface."""
    return (*anchor, (cloth_thickness + TAB_SIZE[2]) / 2)


def attachment_points(points, anchor):
    xy = np.asarray(points)[:, :2]
    distances = np.abs(xy - anchor)
    indices = np.flatnonzero(np.all(distances <= np.asarray(BASE_PATCH) / 2 + 1e-9, axis=1))
    if len(indices) < 4 or np.linalg.matrix_rank(xy[indices] - xy[indices].mean(axis=0)) < 2:
        raise ValueError(f"Tab at {anchor} needs four or more noncollinear attachment nodes")
    return indices.tolist()


def make_tab(index, center, anchor, thickness):
    from isaaclab.sim import CuboidCfg, MassPropertiesCfg, PreviewSurfaceCfg, RigidBodyPropertiesCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg
    from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

    position = tuple(map(float, np.asarray(center) + tab_offset(anchor, thickness)))
    return Object(
        name=f"tape_tab_{index}",
        object_type=ObjectType.RIGID,
        spawner_cfg=CuboidCfg(
            size=TAB_SIZE,
            mass_props=MassPropertiesCfg(mass=TAB_MASS),
            rigid_props=RigidBodyPropertiesCfg(),
            collision_props=PhysxCollisionCfg(contact_offset=0.001, rest_offset=0.0),
            visual_material=PreviewSurfaceCfg(diffuse_color=TAB_COLOR, roughness=0.9),
            physics_material=PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        ),
        initial_pose=Pose(position_xyz=position),
    )


def attach_tab(stage, cloth_path, tab_name, anchor, thickness):
    """Fix a small patch of the simulation mesh to the tab's rigid frame."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    cloth = stage.GetPrimAtPath(cloth_path)
    simulation_mesh = next(
        prim for prim in Usd.PrimRange(cloth)
        if "OmniPhysicsSurfaceDeformableSimAPI" in prim.GetAppliedSchemas()
    )
    points = list(UsdGeom.Mesh(simulation_mesh).GetPointsAttr().Get())
    indices = attachment_points(points, anchor)
    tab_path = Sdf.Path(cloth_path).GetParentPath().AppendChild(tab_name)
    offset = np.asarray(tab_offset(anchor, thickness))
    local = [Gf.Vec3f(*map(float, np.asarray(points[i]) - offset)) for i in indices]
    attachment = stage.DefinePrim(cloth.GetPath().AppendChild("TapeTabAttachment"), "OmniPhysicsVtxXformAttachment")
    attachment.GetRelationship("omniphysics:src0").SetTargets([simulation_mesh.GetPath()])
    attachment.GetRelationship("omniphysics:src1").SetTargets([tab_path])
    attachment.GetAttribute("omniphysics:vtxIndicesSrc0").Set(indices)
    attachment.GetAttribute("omniphysics:localPositionsSrc1").Set(local)
    UsdPhysics.FilteredPairsAPI.Apply(simulation_mesh).CreateFilteredPairsRel().AddTarget(tab_path)
