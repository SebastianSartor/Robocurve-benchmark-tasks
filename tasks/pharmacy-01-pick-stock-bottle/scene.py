"""Author the stock bottles and printed labels on the calibrated YAM station."""

import math

from layout import (
    BODY_HEIGHT, BODY_RADIUS, BOTTLE_HEIGHT, BOTTLE_MASS, CAP_HEIGHT, CAP_RADIUS, CARD_CENTER, CARD_SIZE,
    CONTROL_HZ, SHELF_CENTER, SHELF_SIZE, SLOTS, SLOT_SIZE, ZONE_CENTER, ZONE_SIZE,
)


def texture_mesh(stage, path, vertices, uvs, faces, texture):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*position) for position in vertices])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        [Gf.Vec2f(*uv) for uv in uvs]
    )
    material = UsdShade.Material.Define(stage, path + "/Look")
    surface = UsdShade.Shader.Define(stage, path + "/Look/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.84)
    image = UsdShade.Shader.Define(stage, path + "/Look/Texture")
    image.CreateIdAttr("UsdUVTexture")
    image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture.resolve()))
    image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    image.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    image.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    reader = UsdShade.Shader.Define(stage, path + "/Look/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def print_flat(stage, path, size, texture, z=0.0):
    x, y = (dimension / 2 for dimension in size)
    vertices = [(-x, y, z), (-x, -y, z), (x, -y, z), (x, y, z)]
    texture_mesh(stage, path, vertices, [(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 1, 2, 3)], texture)


def print_front(stage, path, texture):
    radius = BODY_RADIUS + 0.0003
    span = 0.051 / radius
    sections = 32
    points, uvs = [], []
    for index in range(sections + 1):
        u = index / sections
        angle = (u - 0.5) * span
        for v, z in ((0, -0.029), (1, 0.005)):
            points.append((-radius * math.cos(angle), -radius * math.sin(angle), z))
            uvs.append((u, v))
    faces = [(i, i + 2, i + 3, i + 1) for i in range(0, 2 * sections, 2)]
    texture_mesh(stage, path, points, uvs, faces, texture)


def printed_marker(name, xy, size, z, texture):
    from isaaclab.sim import CuboidCfg
    from isaaclab.sim.utils import clone, create_prim, get_current_stage
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        stage = get_current_stage()
        root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
        print_flat(stage, prim_path + "/Print", size, texture)
        return root

    return Object(
        name=name, object_type=ObjectType.BASE, spawner_cfg=CuboidCfg(size=(*size, 0.001), func=spawn),
        initial_pose=Pose(position_xyz=(*xy, z)),
    )


def stock_bottle(index, xy, bottom, labels):
    import isaaclab.sim as sim
    from isaaclab.sim import schemas as schema_ops
    from isaaclab.sim.spawners.shapes import spawn_cylinder
    from isaaclab.sim.utils import bind_physics_material, clone, get_current_stage
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim import schemas
    from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
    from pxr import Gf, UsdGeom

    collision = schemas.PhysxCollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        root = spawn_cylinder(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
        stage = get_current_stage()
        body = UsdGeom.Cylinder.Get(stage, prim_path + "/geometry/mesh")
        body.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, (BODY_HEIGHT - BOTTLE_HEIGHT) / 2))
        cap_path = prim_path + "/Cap"
        cap = UsdGeom.Cylinder.Define(stage, cap_path)
        cap.CreateRadiusAttr(CAP_RADIUS)
        cap.CreateHeightAttr(CAP_HEIGHT)
        cap.CreateAxisAttr("Z")
        cap.AddTranslateOp().Set(Gf.Vec3d(0, 0, (BOTTLE_HEIGHT - CAP_HEIGHT) / 2))
        cap.CreateDisplayColorAttr([(0.91, 0.94, 0.96)])
        schema_ops.define_collision_properties(cap_path, collision, stage=stage)
        bind_physics_material(cap_path, f"{prim_path}/geometry/{cfg.physics_material_path}", stage=stage)
        print_flat(stage, prim_path + "/CapLabel", (0.020, 0.039), labels["cap"], BOTTLE_HEIGHT / 2 + 0.0003)
        print_front(stage, prim_path + "/FrontLabel", labels["front"])
        return root

    return Object(
        name=f"stock_{index}", object_type=ObjectType.RIGID,
        spawner_cfg=sim.CylinderCfg(
            func=spawn, radius=BODY_RADIUS, height=BODY_HEIGHT, collision_props=collision,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.84, 0.86, 0.91), roughness=0.55),
            physics_material=PhysxRigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
            rigid_props=schemas.PhysxRigidBodyPropertiesCfg(), mass_props=sim.MassPropertiesCfg(mass=BOTTLE_MASS),
        ),
        initial_pose=Pose(position_xyz=(*xy, bottom + BOTTLE_HEIGHT / 2 + 0.001)),
    )


def make_station(layout, textures, *, rig, wrist_cam_mount, device):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import _box, env_cfg_callback, task_room
    from yam_arena.station import TABLE_TOP

    top = TABLE_TOP + SHELF_SIZE[2]
    shelf = _box("stock_shelf", SHELF_SIZE, (*SHELF_CENTER, TABLE_TOP + SHELF_SIZE[2] / 2), (0.25, 0.31, 0.40))
    fixtures = [shelf, printed_marker("order_card", CARD_CENTER, CARD_SIZE, TABLE_TOP + 0.001, textures["order"]),
                printed_marker("staging_zone", ZONE_CENTER, ZONE_SIZE, TABLE_TOP + 0.001, textures["staging"])]
    bottles = []
    for i, (xy, stock) in enumerate(zip(SLOTS, layout.bottles, strict=True)):
        fixtures.append(printed_marker(f"slot_mark_{i}", xy, SLOT_SIZE, top + 0.0006, textures["slot"]))
        bottles.append(stock_bottle(i, xy, top, textures[stock.key]))

    robot = BimanualYamEmbodiment(
        enable_cameras=True, wrist_cam_mount=wrist_cam_mount, contact_sensors=False, rig=rig,
    )
    arena = IsaacLabArenaEnvironment(
        name="pharmacy-01-pick-stock-bottle",
        scene=Scene([*task_room("station", rig.top_cam), *fixtures, *bottles]),
        embodiment=robot,
        env_cfg_callback=env_cfg_callback(CONTROL_HZ),
    )
    return ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=1, device=device, solve_relations=False)).make_registered()
