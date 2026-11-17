"""Preview or run textile-04-position-repair-patch on the YAM station."""

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from layout import CONTROL_HZ, MAX_STEPS, make_layout
from mesh import task_meshes


TASK_NAME = "textile-04-position-repair-patch-tape-tab"
RELEASE_CLEARANCE = 0.04


def make_robot(layout, *, live, setup_image):
    import numpy as np
    from inspect_robots.embodiment import RESETTABLE, SEEDABLE
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
    from yam_arena.inspect_embodiment import (
        ARMS, DESKTOP_DISPLAY, DESKTOP_SIZE, JOINT_GUIDE, YamArenaEmbodiment, window_display,
    )

    from scoring import measure, verify_start
    from setup_scene import build_scene

    meshes = task_meshes()

    class TextileEmbodiment(YamArenaEmbodiment):
        def __init__(self):
            super().__init__(
                scene="station", object_jitter=0, control_hz=CONTROL_HZ,
                cam_width=320, cam_height=176, headless=not live,
            )
            self.app = None
            self.info = replace(
                self.info,
                capabilities=frozenset({SEEDABLE, RESETTABLE}),
                supported_target_kinds=frozenset(),
                docs=JOINT_GUIDE,
                environment_id=f"{self.info.environment_id}-{TASK_NAME}",
            )

        def _ensure_sim(self, goal):
            if self._env is not None:
                return
            from isaaclab.app import AppLauncher

            options = dict(headless=self.headless, device=self.device, enable_cameras=True)
            if not self.headless:
                display = window_display(os.environ)
                os.environ["DISPLAY"] = display
                options["visualizer"] = ["kit"]
                if display == DESKTOP_DISPLAY:
                    options["window_width"], options["window_height"] = DESKTOP_SIZE
            self.app = AppLauncher(**options).app

            import torch

            self._env = build_scene(
                TASK_NAME, meshes, rig=self.rig, wrist_cam_mount=self.wrist_cam_mount, device=self.device,
            )
            self._base = self._env.unwrapped
            self._torch = torch
            self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
            self._ids = {
                name: (
                    self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                    self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0],
                )
                for name in ARMS
            }

        def panel_nodes(self, index):
            return self._base.scene[f"panel_{index}"].data.nodal_pos_w.torch[0].cpu().numpy()

        def tab_pose(self, index):
            pose = self._base.scene[f"tape_tab_{index}"].data.root_link_pose_w
            return pose.torch[0].cpu().numpy().copy()

        def reset(self, scene, *, seed=None):
            observation = super().reset(scene, seed=seed)
            if setup_image is not None:
                self.save_setup(setup_image)
            if not self.headless:
                self._base.sim.set_camera_view((0.63, -0.39, 1.28), (0.17, 0.04, 0.75))
            return observation

        def observe_parked(self):
            # The final score must reflect where the policy left its hands.
            return None

        def cloth_nodes(self):
            nodes = [self.panel_nodes(index) for index in range(len(meshes))]
            if any(panel.shape != rest.shape or not np.isfinite(panel).all()
                   for panel, (rest, _) in zip(nodes, meshes)):
                raise RuntimeError("Unexpected simulation cloth topology or non-finite nodes")
            return nodes

        def released_panels(self, nodes):
            from yam_arena.ik import tcp_pose
            from yam_arena.station import BASES

            hands = []
            for side in ("left", "right"):
                name = f"{side}_robot"
                joints, _ = self._ids[name]
                angles = self._base.scene[name].data.joint_pos.torch[0, joints].cpu().numpy()
                hands.append(tcp_pose(angles, BASES[side])[0])
            grasp_regions = (nodes[0], np.vstack((nodes[1], self.tab_pose(1)[:3])))
            return tuple(
                bool(np.linalg.norm(panel[:, None, :] - hands, axis=2).min() > RELEASE_CLEARANCE)
                for panel in grasp_regions
            )

        def measure(self):
            nodes = self.cloth_nodes()
            return measure(nodes, self.released_panels(nodes))

        def save_setup(self, path):
            from PIL import Image

            rgb = self._base.scene["top_cam"].data.output["rgb"]
            rgb = rgb.torch if hasattr(rgb, "torch") else rgb
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb[0, ..., :3].cpu().numpy()).save(path)
            path.with_suffix(".json").write_text(json.dumps(layout.describe(), indent=2) + "\n")

        def verify_settled(self):
            from yam_arena.station import TABLE_TOP

            for _ in range(40):
                self._env.step(self._ready)
            nodes = self.cloth_nodes()
            result = measure(nodes, self.released_panels(nodes))
            verify_start(result, nodes, TABLE_TOP)
            return result

        def hold_view(self, seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and self._base.sim.is_headless_or_exist_active_visualizer():
                self._base.sim.render()
                time.sleep(0.1)

    return TextileEmbodiment()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Reproducibility seed for the run.")
    parser.add_argument("--layout-only", action="store_true", help="Print the layout on the Mac without Isaac Sim.")
    parser.add_argument("--dream", action="store_true", help="Run Dream-1. Otherwise preview the starting scene.")
    parser.add_argument("--live", action="store_true", help="Show the Isaac Sim viewport on Brev.")
    parser.add_argument("--hold-open", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--setup-image", type=Path, help="Save the initial top-camera PNG and layout JSON.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not 1 <= args.max_steps <= MAX_STEPS:
        parser.error(f"--max-steps must be between 1 and {MAX_STEPS}")
    if args.dream and args.log_dir is None:
        parser.error("--dream requires --log-dir")
    if args.hold_open < 0 or (args.hold_open and not args.live):
        parser.error("--hold-open requires --live and a nonnegative number of seconds")

    if args.setup_image is not None and args.setup_image.suffix.lower() != ".png":
        parser.error("--setup-image must end in .png")

    layout = make_layout(args.seed)
    print(layout.instruction, flush=True)
    if args.layout_only:
        print(json.dumps(layout.describe(), indent=2))
        return 0

    from inspect_robots import Scene, Task, eval
    from inspect_robots.scorer import Score

    class TextileScorer:
        name = TASK_NAME

        def __call__(self, record, target):
            measured = record.metadata[TASK_NAME]
            return Score(value=measured["score"], metadata=measured)

    scene = Scene(id=f"layout-{args.seed}", instruction=layout.instruction)
    robot = make_robot(layout, live=args.live, setup_image=args.setup_image)
    status = 1
    try:
        if args.dream:
            def capture_score(record, scene):
                record.metadata[TASK_NAME] = asdict(robot.measure())

            task = Task(name=TASK_NAME, scenes=[scene], scorer=TextileScorer(), max_steps=args.max_steps)
            [log] = eval(
                task, "dream", robot, seed=args.seed, log_dir=str(args.log_dir),
                fail_on_error=True, guard=["clamp", "delta-limit"], store_frames=True, before_scoring=capture_score,
            )
            for sample in log.samples:
                print(f"{sample.scene_id}: {sample.status}; {sample.reduced}", flush=True)
            status = 0 if log.status == "success" else 1
        else:
            robot.reset(scene, seed=args.seed)
            result = robot.verify_settled()
            print(f"Starting scene verified: {result}", flush=True)
            status = 0
        if args.hold_open:
            robot.hold_view(args.hold_open)
    except KeyboardInterrupt:
        status = 130
    except Exception:
        traceback.print_exc()
    finally:
        if robot.app is not None:
            robot.app.close(exit_code=status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
