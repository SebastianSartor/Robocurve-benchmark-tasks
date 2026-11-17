"""Run Dream-1 on the TextileBench cloth scene and record the policy cameras."""

import argparse
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
from inspect_robots import Scene, Task, eval
from inspect_robots.scorer import Score
from inspect_robots.embodiment import RESETTABLE, SEEDABLE
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import (
    ARMS, DESKTOP_DISPLAY, DESKTOP_SIZE, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment, window_display,
)

from cloth_scoring import sort_result
from cloth_station_preview import INSTRUCTION, SLOTS, ZONE_SIZE, ZONES, make_station
from cloth_swatch_probe import make_mesh


class ClothStationEmbodiment(YamArenaEmbodiment):
    """Reuse YAM's joint controls and camera contract with a deformable Arena scene."""

    def __init__(self, layout_seed, *, live=False):
        super().__init__(
            scene="station", object_jitter=0, control_hz=20, cam_width=320, cam_height=176, headless=not live,
        )
        self.layout_seed = layout_seed
        self.app = None
        self.info = replace(
            self.info,
            capabilities=frozenset({SEEDABLE, RESETTABLE}),
            supported_target_kinds=frozenset(),
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)),
            environment_id=f"{self.info.environment_id}-textile-07-sort-damaged-fabric-tape-tabs",
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

        self._env, self.damaged_slot = make_station(
            self.layout_seed, rig=self.rig, wrist_cam_mount=self.wrist_cam_mount,
            control_hz=self.info.control_hz, device=self.device,
        )
        self._base = self._env.unwrapped
        self.faces = [make_mesh(slot == self.damaged_slot).faces for slot in range(2)]
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
            self._base.sim.set_camera_view((0.72, -0.55, 1.43), (0.25, 0.0, 0.76))
            print("Isaac Sim viewport ready.", flush=True)
        return observation

    def panel_nodes(self, index):
        return self._base.scene[f"panel_{index}"].data.nodal_pos_w.torch[0].cpu().numpy()

    def tab_pose(self, index):
        pose = self._base.scene[f"tape_tab_{index}"].data.root_link_pose_w
        return pose.torch[0].cpu().numpy().copy()

    def observe_parked(self):
        return None

    def measure(self):
        from yam_arena.ik import tcp_pose
        from yam_arena.station import BASES

        hands = []
        for side in ("left", "right"):
            name = f"{side}_robot"
            joint_ids, _ = self._ids[name]
            angles = self._base.scene[name].data.joint_pos.torch[0, joint_ids].cpu().numpy()
            hands.append(tcp_pose(angles, BASES[side])[0])
        nodes = [self.panel_nodes(slot) for slot in range(2)]
        tabs = [self.tab_pose(slot)[:3] for slot in range(2)]
        return sort_result(nodes, self.faces, np.asarray(hands), self.damaged_slot, ZONES, ZONE_SIZE, tabs=tabs)

    def verify_settled(self):
        from yam_arena.station import TABLE_TOP

        for _ in range(40):
            self._env.step(self._ready)
        for slot, expected in enumerate(SLOTS):
            nodes = self.panel_nodes(slot)
            heights = nodes[:, 2] - TABLE_TOP
            if (not np.isfinite(nodes).all() or np.linalg.norm(nodes[:, :2].mean(0) - expected) > 0.01
                    or heights.min() < -0.005 or heights.max() > 0.014):
                raise RuntimeError(f"Sorting panel {slot} did not settle flat at its starting location")
        result = self.measure()
        if any(panel["fraction_in_zone"] > 1e-6 for panel in result["panels"].values()):
            raise RuntimeError("A sorting panel started inside its destination")
        return result

    def hold_view(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and self._base.sim.is_headless_or_exist_active_visualizer():
            self._base.sim.render()
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Seed for the initial damaged-panel position.")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum control steps at 20 Hz.")
    parser.add_argument("--live", action="store_true", help="Show the run in the Brev desktop's Isaac Sim window.")
    parser.add_argument("--hold-open", type=int, default=0, help="Seconds to show the final scene after a live run.")
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 3600:
        parser.error("--max-steps must be between 1 and 3600")
    if args.hold_open < 0 or (args.hold_open and not args.live):
        parser.error("--hold-open must be nonnegative and requires --live")

    robot = ClothStationEmbodiment(args.seed, live=args.live)
    def capture_score(record, scene):
        record.metadata["cloth_sort"] = robot.measure()

    class SortScorer:
        name = "cloth_sort"

        def __call__(self, record, target):
            result = record.metadata[self.name]
            return Score(value=result["score"], metadata=result)

    task = Task(
        name="textile-07-sort-damaged-fabric-tape-tabs",
        scenes=[Scene(id=f"layout-{args.seed}", instruction=INSTRUCTION)],
        scorer=SortScorer(),
        max_steps=args.max_steps,
    )
    status = 1
    try:
        print(f"Dream-1 instruction: {INSTRUCTION}", flush=True)
        [log] = eval(
            task, "dream", robot, seed=args.seed, log_dir=str(args.log_dir),
            fail_on_error=True, guard=["clamp", "delta-limit"], store_frames=True,
            before_scoring=capture_score,
        )
        for sample in log.samples:
            print(f"{sample.scene_id}: {sample.status}; {sample.reduced}; {sample.termination_reasons}", flush=True)
            if sample.error:
                print(sample.error, flush=True)
        print(f"Run status: {log.status}. Logs and camera frames: {args.log_dir}", flush=True)
        status = 0 if log.status == "success" else 1
        if status == 0 and args.hold_open:
            print(f"Showing the final scene for {args.hold_open} seconds.", flush=True)
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
