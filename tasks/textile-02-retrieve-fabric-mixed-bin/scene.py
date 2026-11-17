"""Spawn deformable cloth at the Isaac Lab Arena YAM station."""

import numpy as np

from cloth import ASSETS, CONTROL_HZ, PANEL_SIZE, panel_mesh


# Thickness in m, density in kg/m³, and bend stiffness.
# Density and bending coefficients preserve the earlier areal mass and flex.
MATERIALS = {
    "cotton": (0.0038, 189.5, 9.84e4),
    "linen": (0.0033, 189.4, 1.48e5),
    "felt": (0.0048, 150.0, 8.68e4),
}


def _bind_texture(stage, mesh, texture_path, uvs):
    from pxr import Sdf, UsdGeom, UsdShade, Vt

    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        Vt.Vec2fArray.FromNumpy(uvs.astype(np.float32))
    )
    path = mesh.GetPath().GetParentPath().AppendChild("FabricAppearance")
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path.AppendChild("Surface"))
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.93)
    texture = UsdShade.Shader.Define(stage, path.AppendChild("Texture"))
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    reader = UsdShade.Shader.Define(stage, path.AppendChild("UV"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def panel_spawner(fabric, *, tab_name, anchor):
    from tape_tab import attach_tab

    import trimesh
    from isaaclab.sim import MeshRectangleCfg
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg
    from pxr import UsdGeom, Vt

    thickness, density, bend = MATERIALS[fabric.material]
    vertices, faces = panel_mesh()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, **kwargs)
        stage = get_current_stage()
        visual = UsdGeom.Mesh.Get(stage, f"{prim_path}/geometry/mesh")
        top = vertices + (0, 0, thickness / 2)
        bottom = vertices - (0, 0, thickness / 2)
        count = len(vertices)
        _, edge_ids, edge_counts = np.unique(mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True)
        a, b = mesh.edges[edge_counts[edge_ids] == 1].T
        shell = np.vstack((faces, faces[:, ::-1] + count,
                           np.column_stack((b, a, a + count)), np.column_stack((b, a + count, b + count))))
        points = Vt.Vec3fArray.FromNumpy(np.vstack((top, bottom)).astype(np.float32))
        visual.GetPointsAttr().Set(points)
        visual.GetFaceVertexCountsAttr().Set([3] * len(shell))
        visual.GetFaceVertexIndicesAttr().Set(shell.ravel().tolist())
        visual.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        visual.GetPrim().GetAttribute("deformablePose:default:omniphysics:points").Set(points)
        uvs = np.tile(vertices[:, :2] / PANEL_SIZE + 0.5, (2, 1))
        _bind_texture(stage, visual, ASSETS / f"{fabric.key}.png", uvs)
        attach_tab(stage, prim_path, tab_name, anchor, thickness)
        return stage.GetPrimAtPath(prim_path)

    return MeshRectangleCfg(
        func=spawn,
        size=PANEL_SIZE,
        edge_refinement=1.0,
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        collision_props=[PhysxCollisionCfg(rest_offset=thickness / 2, contact_offset=0.005)],
        physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
            density=density, surface_thickness=thickness, youngs_modulus=4.0e5, poissons_ratio=0.25,
            surface_bend_stiffness=bend, elasticity_damping=0.1, bend_damping=0.1,
            static_friction=0.7, dynamic_friction=0.5,
        ),
    )


def make_station(name, panels, fixtures, *, rig, wrist_cam_mount, device):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room

    robot = BimanualYamEmbodiment(
        enable_cameras=True, wrist_cam_mount=wrist_cam_mount, contact_sensors=False, rig=rig,
    )
    arena = IsaacLabArenaEnvironment(
        name=name,
        scene=Scene([*task_room("station", rig.top_cam), *fixtures, *panels]),
        embodiment=robot,
        env_cfg_callback=env_cfg_callback(CONTROL_HZ),
    )
    config = ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)
    return ArenaEnvBuilder(arena, config).make_registered()
