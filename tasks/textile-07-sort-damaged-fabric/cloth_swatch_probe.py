"""Drop TextileBench swatches onto ledges using the Brev image's PhysX backend."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import trimesh


SWATCH_SIZE = (0.140, 0.100)
HOLE_SIZE = 0.025
CLOTH_THICKNESS = 0.005
CLOTH_TEXTURE = Path(__file__).resolve().with_name("terry_gingham.png")
DROP_Z = 0.095
LEDGE_TOP = 0.060


def make_mesh(damaged):
    half_hole = HOLE_SIZE / 2

    def grid(half_width):
        # Put both cutout edges on grid lines, with mesh edges at most 6.25 mm long along each axis.
        return np.concatenate((
            np.linspace(-half_width, -half_hole, 11)[:-1],
            np.linspace(-half_hole, half_hole, 5)[:-1],
            np.linspace(half_hole, half_width, 11),
        ))

    xs, ys = (grid(length / 2) for length in SWATCH_SIZE)
    vertices = [(x, y, 0.0) for y, x in product(ys, xs)]
    faces = []
    for j, i in product(range(len(ys) - 1), range(len(xs) - 1)):
        inside_hole = (
            abs((xs[i] + xs[i + 1]) / 2) < half_hole
            and abs((ys[j] + ys[j + 1]) / 2) < half_hole
        )
        if damaged and inside_hole:
            continue
        a = j * len(xs) + i
        b = a + 1
        d = a + len(xs)
        c = d + 1
        faces.extend(((a, b, c), (a, c, d)))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    expected_area = SWATCH_SIZE[0] * SWATCH_SIZE[1] - (HOLE_SIZE**2 if damaged else 0)
    if not np.isclose(mesh.area, expected_area):
        raise RuntimeError(f"Incorrect swatch area: {mesh.area} m², expected {expected_area} m²")
    return mesh


def apply_cloth_finish(stage, prim_path, mesh):
    from pxr import Sdf, UsdGeom, UsdShade, Vt

    # The spawner has already copied the flat mesh to sim_mesh. Give only the
    # skinned render mesh two faces and walls around the perimeter and cutout.
    visual = UsdGeom.Mesh.Get(stage, f"{prim_path}/geometry/mesh")
    vertex_count = len(mesh.vertices)
    top = mesh.vertices.copy()
    top[:, 2] += CLOTH_THICKNESS / 2
    bottom = mesh.vertices.copy()
    bottom[:, 2] -= CLOTH_THICKNESS / 2
    points = np.vstack((top, bottom))
    _, edge_ids, edge_counts = np.unique(mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True)
    a, b = mesh.edges[edge_counts[edge_ids] == 1].T
    faces = np.vstack((
        mesh.faces,
        mesh.faces[:, ::-1] + vertex_count,
        np.column_stack((b, a, a + vertex_count)),
        np.column_stack((b, a + vertex_count, b + vertex_count)),
    ))

    usd_points = Vt.Vec3fArray.FromNumpy(points.astype(np.float32))
    visual.GetPointsAttr().Set(usd_points)
    visual.GetFaceVertexCountsAttr().Set([3] * len(faces))
    visual.GetFaceVertexIndicesAttr().Set(faces.ravel().tolist())
    visual.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    visual.GetPrim().GetAttribute("deformablePose:default:omniphysics:points").Set(usd_points)
    uv = Vt.Vec2fArray.FromNumpy((points[:, :2] / SWATCH_SIZE + 0.5).astype(np.float32))
    UsdGeom.PrimvarsAPI(visual).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(uv)

    material = UsdShade.Material.Define(stage, f"{prim_path}/geometry/TerryCotton")
    surface = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.93)
    texture = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(CLOTH_TEXTURE))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    reader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)


def swatch_spawner(damaged, *, tape_tab=None):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

    mesh = make_mesh(damaged)

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        # Use the installed MeshRectangle spawner's mesh path with our cutout topology.
        _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, **kwargs)
        stage = get_current_stage()
        apply_cloth_finish(stage, prim_path, mesh)
        if tape_tab is not None:
            from tape_tab import attach_tab

            tab_name, anchor = tape_tab
            attach_tab(stage, prim_path, tab_name, anchor, CLOTH_THICKNESS)
        return stage.GetPrimAtPath(prim_path)

    return sim_utils.MeshRectangleCfg(
        func=spawn,
        size=SWATCH_SIZE,
        edge_refinement=1.0,
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        collision_props=[PhysxCollisionCfg(rest_offset=CLOTH_THICKNESS / 2, contact_offset=0.005)],
        physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
            # Preserve the original mass per area as thickness increases.
            density=200.0,
            surface_thickness=CLOTH_THICKNESS,
            poissons_ratio=0.25,
            youngs_modulus=4.0e5,
            # PhysX scales edge bending by thickness cubed. Keep the thicker cloth flexible.
            surface_bend_stiffness=8.19e4,
            elasticity_damping=0.1,
            bend_damping=0.1,
            static_friction=0.7,
            dynamic_friction=0.5,
        ),
    )


def spawn_swatch(name, x, damaged):
    from isaaclab.assets import DeformableObjectCfg

    mesh = make_mesh(damaged)
    cfg = DeformableObjectCfg(
        prim_path=f"/World/{name}",
        spawn=swatch_spawner(damaged),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(x, 0.0, DROP_Z)),
    )
    print(f"{name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles, "
          f"{mesh.area * 1e6:.0f} mm² rest area", flush=True)
    return cfg.class_type(cfg)


def run_probe(args):
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxCfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg()))
    sim.set_camera_view((0.30, -0.43, 0.35), (0.0, 0.0, 0.05))

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2400.0)
    light.func("/World/Light", light)
    ledge = sim_utils.CuboidCfg(
        size=(0.155, 0.075, LEDGE_TOP),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.32)),
    )
    cloths = {}
    for name, x, damaged in (("Intact", -0.09, False), ("Damaged", 0.09, True)):
        ledge.func(f"/World/{name}Ledge", ledge, translation=(x, 0.028, LEDGE_TOP / 2))
        cloths[name] = spawn_swatch(name, x, damaged)

    sim.reset()
    while True:
        for cloth in cloths.values():
            cloth.write_nodal_state_to_sim_index(cloth.data.default_nodal_state_w.torch.clone())
            cloth.reset()
        for _ in range(args.steps):
            if not sim.is_headless_or_exist_active_visualizer():
                return
            sim.step()
            for cloth in cloths.values():
                cloth.update(sim.get_physics_dt())

        for name, cloth in cloths.items():
            z = cloth.data.nodal_pos_w.torch[0, :, 2]
            if not z.isfinite().all():
                raise RuntimeError(f"{name} has non-finite cloth node heights")
            low, high = float(z.min()), float(z.max())
            print(f"{name}: final node heights {low * 1000:.1f}–{high * 1000:.1f} mm", flush=True)
            if low < -0.005 or abs(high - LEDGE_TOP) > 0.012 or high - low < 0.008:
                raise RuntimeError(f"{name} did not settle with a measurable drape over the ledge")
        print("PASS: both swatches draped over their ledges.", flush=True)
        if not args.repeat:
            return


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240, help="Physics steps per drop.")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer.")
    parser.add_argument("--repeat", action="store_true", help="Repeat the drop while the viewer is open.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or (args.headless and args.repeat):
        parser.error("--steps must be positive and --repeat requires a viewer")

    # Boot Kit before importing simulation modules, as the Brev YAM runner does.
    launcher = AppLauncher(headless=args.headless, device=args.device, visualizer=[] if args.headless else ["kit"])
    run_probe(args)
    launcher.app.close()


if __name__ == "__main__":
    main()
