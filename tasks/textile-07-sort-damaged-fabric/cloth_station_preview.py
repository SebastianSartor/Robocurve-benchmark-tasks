"""Preview TextileBench cloth sorting geometry on the bimanual YAM station."""

import argparse
from pathlib import Path

import numpy as np

from cloth_swatch_probe import CLOTH_THICKNESS, swatch_spawner
from tape_tab import make_tab


ZONE_SIZE = (0.180, 0.140)
SLOTS = ((0.17, 0.12), (0.17, -0.12))
TAPE_ANCHOR = (-0.044, 0.027)
ZONES = {
    "Damaged": ((0.38, 0.12), (199, 57, 62)),
    "Undamaged": ((0.38, -0.12), (44, 142, 76)),
}
INSTRUCTION = (
    "Put the cloth with the square hole fully inside the red zone labeled Damaged. "
    "Put the cloth without a hole fully inside the green zone labeled Undamaged. "
    "Use the raised tape tabs to pick up the cloths. Release both cloths inside their zones."
)


def zone_marker(label, xy, color, table_z):
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Sdf, UsdGeom, UsdShade

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone, create_prim, get_current_stage
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    texture_path = Path(__file__).resolve().with_name(f"zone_{label.lower()}.png")
    texture = Image.new("RGB", (560, 720), color)
    draw = ImageDraw.Draw(texture)
    draw.rectangle((0, 0, 559, 719), outline=(41, 40, 35), width=8)
    draw.text((280, 48), label, font=ImageFont.load_default(size=54), fill=(255, 250, 236), anchor="mm")
    texture.save(texture_path)

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        stage = get_current_stage()
        root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
        x, y = (length / 2 for length in ZONE_SIZE)
        plane = UsdGeom.Mesh.Define(stage, f"{prim_path}/Marking")
        plane.CreatePointsAttr([(-x, y, 0), (-x, -y, 0), (x, -y, 0), (x, y, 0)])
        plane.CreateFaceVertexCountsAttr([4])
        plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        plane.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(plane).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
            [(0, 0), (1, 0), (1, 1), (0, 1)]
        )

        material = UsdShade.Material.Define(stage, f"{prim_path}/LabelMaterial")
        surface = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        tex = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        uv = UsdShade.Shader.Define(stage, f"{material.GetPath()}/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material)
        return root

    return Object(
        name=f"zone_{label.lower()}",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(size=(*ZONE_SIZE, 0.001), func=spawn),
        initial_pose=Pose(position_xyz=(*xy, table_z + 0.0005)),
    )


def make_station(seed, *, rig=None, wrist_cam_mount="link", control_hz=20, device="cuda:0", enable_cameras=True):
    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from yam_arena import station
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room

    rig = rig if rig is not None else station.rig("calibrated")
    robot = BimanualYamEmbodiment(
        enable_cameras=enable_cameras, wrist_cam_mount=wrist_cam_mount, contact_sensors=False, rig=rig
    )
    damaged_slot = int(np.random.default_rng(seed).integers(2))
    spawn_height = station.TABLE_TOP + 0.012
    tabs = [make_tab(i, (*xy, spawn_height), TAPE_ANCHOR, CLOTH_THICKNESS) for i, xy in enumerate(SLOTS)]
    panels = [
        DeformableObject(
            name=f"panel_{i}",
            spawner_cfg=swatch_spawner(damaged=i == damaged_slot, tape_tab=(tabs[i].name, TAPE_ANCHOR)),
            initial_pose=Pose(position_xyz=(*xy, spawn_height)),
        )
        for i, xy in enumerate(SLOTS)
    ]
    zones = [zone_marker(label, xy, color, station.TABLE_TOP) for label, (xy, color) in ZONES.items()]
    arena = IsaacLabArenaEnvironment(
        name="textile-07-sort-damaged-fabric-tape-tabs",
        scene=Scene([*task_room("station", rig.top_cam), *tabs, *panels, *zones]),
        embodiment=robot,
        env_cfg_callback=env_cfg_callback(control_hz),
    )
    config = ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)
    return ArenaEnvBuilder(arena, config).make_registered(), damaged_slot


def run_preview(args):
    import torch
    from yam_arena.constants import READY
    from yam_arena.station import TABLE_TOP

    env, damaged_slot = make_station(args.seed)
    try:
        env.reset(seed=args.seed)
        base = env.unwrapped
        ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=base.device)
        for _ in range(30):
            env.step(ready)
        for i in range(len(SLOTS)):
            cloth = base.scene[f"panel_{i}"]
            nodes = cloth.data.nodal_pos_w.torch[0]
            expected_nodes = 616 if i == damaged_slot else 625
            if len(nodes) != expected_nodes or not torch.isfinite(nodes).all():
                raise RuntimeError(f"Unexpected simulation nodes for panel {i}")
            height = float(nodes[:, 2].mean())
            if abs(height - TABLE_TOP) > 0.008:
                raise RuntimeError(f"Panel {i} did not settle on the YAM table: z={height:.4f} m")
            print(f"panel_{i}: {len(nodes)} simulated vertices, mean z={height:.4f} m", flush=True)
        print(f"PASS: both panels settled on the YAM table. Damaged panel is in slot {damaged_slot}.", flush=True)
        if not args.headless:
            base.sim.set_camera_view((0.72, -0.55, 1.43), (0.25, 0.0, 0.76))
            print("Close the Isaac Sim window or press Ctrl+C to exit.", flush=True)
            while base.sim.is_headless_or_exist_active_visualizer():
                env.step(ready)
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Seed for the initial damaged-panel position.")
    parser.add_argument("--headless", action="store_true", help="Verify the scene without opening a viewer.")
    args = parser.parse_args()
    launcher = AppLauncher(
        headless=args.headless, device="cuda:0", enable_cameras=True, visualizer=[] if args.headless else ["kit"]
    )
    try:
        run_preview(args)
    except KeyboardInterrupt:
        pass
    launcher.app.close()


if __name__ == "__main__":
    main()
