"""Preview, score, and record PharmacyBench Task 01 on the two-arm YAM station."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback

from layout import CONTROL_HZ, INSTRUCTION, MAX_STEPS, SLOTS, make_layout


TASK_NAME = "pharmacy-01-pick-stock-bottle"


def make_robot(layout, textures, *, live):
    import numpy as np
    from inspect_robots.embodiment import RESETTABLE, SEEDABLE
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
    from yam_arena.inspect_embodiment import (
        ARMS, DESKTOP_DISPLAY, DESKTOP_SIZE, JOINT_GUIDE, YamArenaEmbodiment, window_display,
    )

    from scene import make_station
    from scoring import read_score

    class PharmacyEmbodiment(YamArenaEmbodiment):
        def __init__(self):
            super().__init__(
                scene="station", object_jitter=0, control_hz=CONTROL_HZ,
                cam_width=320, cam_height=176, headless=not live,
            )
            self.app = None
            self.info = replace(
                self.info, capabilities=frozenset({RESETTABLE, SEEDABLE}), environment_id=TASK_NAME,
                supported_target_kinds=frozenset(), docs=JOINT_GUIDE,
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

            self._env = make_station(
                layout, textures, rig=self.rig, wrist_cam_mount=self.wrist_cam_mount, device=self.device,
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

        def reset(self, scene, *, seed=None):
            observation = super().reset(scene, seed=seed)
            if not self.headless:
                self._base.sim.set_camera_view((0.72, -0.44, 1.38), (0.23, 0.0, 0.77))
            return observation

        def observe_parked(self):
            return None

        def bottle_state(self, index):
            rigid = self._base.scene[f"stock_{index}"].data
            return (rigid.root_link_pose_w.torch[0].cpu().numpy().copy(),
                    rigid.root_link_vel_w.torch[0].cpu().numpy().copy())

        def measure(self):
            from yam_arena.ik import tcp_pose
            from yam_arena.station import BASES, TABLE_TOP

            hands = []
            for side in ("left", "right"):
                name = f"{side}_robot"
                joints, _ = self._ids[name]
                angles = self._base.scene[name].data.joint_pos.torch[0, joints].cpu().numpy()
                hands.append(tcp_pose(angles, BASES[side])[0])
            poses, velocities = zip(*(self.bottle_state(i) for i in range(len(SLOTS))), strict=True)
            return read_score(layout, np.asarray(poses), np.asarray(velocities), np.asarray(hands), TABLE_TOP)

        def frame(self, camera):
            rgb = self._base.scene[camera].data.output["rgb"]
            rgb = rgb.torch if hasattr(rgb, "torch") else rgb
            return rgb[0, ..., :3].cpu().numpy()

        def command(self, actions):
            tensor = self._torch.as_tensor(actions, dtype=self._torch.float32, device=self._base.device)[None, :]
            _, _, terminated, truncated, _ = self._env.step(tensor)
            if (terminated | truncated).any():
                raise RuntimeError("Isaac Lab reset during the demonstration")

    return PharmacyEmbodiment()


def preview(robot, layout, output):
    import numpy as np
    from PIL import Image
    from yam_arena.station import TABLE_TOP

    from layout import BOTTLE_HEIGHT, SHELF_SIZE

    for _ in range(20):
        robot._env.step(robot._ready)
    for i, xy in enumerate(SLOTS):
        pose, velocity = robot.bottle_state(i)
        expected = (*xy, TABLE_TOP + SHELF_SIZE[2] + BOTTLE_HEIGHT / 2)
        if np.linalg.norm(pose[:3] - expected) > 0.004 or np.linalg.norm(velocity[:3]) > 0.01:
            raise RuntimeError(f"Bottle {i} did not settle in its slot: {pose}")
    result = robot.measure()
    if result.score != 0 or result.displaced_distractors:
        raise RuntimeError(f"Invalid starting state: {result}")
    (output / "layout.json").write_text(json.dumps({**layout.describe(), "start_score": asdict(result)}, indent=2) + "\n")
    Image.fromarray(robot.frame("top_cam")).save(output / "robot-camera.png")
    camera = robot._base.scene["top_cam"]
    for name, eye, target in (
        ("setup", (0.068, 0, 0.44), (0.24, 0.0, 0.015)),
        ("bottle-labels", (0.175, -0.107, 0.225), (0.300, -0.106, 0.042)),
        ("prescription", (0.072, 0.186, 0.183), (0.155, 0.186, 0.001)),
    ):
        camera.set_world_poses_from_view(
            np.asarray([eye]) + (0, 0, TABLE_TOP), np.asarray([target]) + (0, 0, TABLE_TOP),
        )
        for _ in range(3):
            robot._env.step(robot._ready)
        Image.fromarray(robot.frame("top_cam")).save(output / f"{name}.png")
    print(f"Starting scene verified: {result}", flush=True)


def dream(robot, scene, seed, max_steps, output):
    from inspect_robots import Task, eval
    from inspect_robots.scorer import Score

    def capture(record, _scene):
        record.metadata[TASK_NAME] = asdict(robot.measure())

    class PharmacyScorer:
        name = TASK_NAME

        def __call__(self, record, target):
            reading = record.metadata[self.name]
            return Score(value=reading["score"], metadata=reading)

    task = Task(name=TASK_NAME, scenes=[scene], scorer=PharmacyScorer(), max_steps=max_steps)
    [log] = eval(task, "dream", robot, seed=seed, log_dir=str(output), fail_on_error=True,
                 store_frames=True, guard=["clamp", "delta-limit"], before_scoring=capture)
    for sample in log.samples:
        print(sample.status, sample.reduced, flush=True)
    return 0 if log.status == "success" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dream", action="store_true")
    modes.add_argument("--expert", action="store_true")
    modes.add_argument("--layout-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= MAX_STEPS:
        parser.error(f"--max-steps must be 1 through {MAX_STEPS}")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    layout = make_layout(args.seed)
    if args.layout_only:
        print(json.dumps(layout.describe(), indent=2))
        return 0
    if args.output is None:
        parser.error("name a run directory with --output")

    from inspect_robots import Scene
    from labels import prepare

    args.output.mkdir(parents=True, exist_ok=True)
    scene = Scene(id=f"layout-{args.seed}", instruction=INSTRUCTION)
    status = 1
    with TemporaryDirectory(prefix="pharmacy-stock-labels-") as directory:
        robot = make_robot(layout, prepare(layout, directory), live=args.live)
        try:
            if args.dream:
                status = dream(robot, scene, args.seed, args.max_steps, args.output)
            else:
                robot.reset(scene, seed=args.seed)
                if args.expert:
                    from expert import record_expert

                    status = record_expert(robot, layout, args.output, args.max_steps)
                else:
                    preview(robot, layout, args.output)
                    status = 0
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
