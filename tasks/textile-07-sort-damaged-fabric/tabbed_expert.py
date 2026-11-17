"""Record and score physical YAM demonstrations using the cloth tape tabs."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import traceback

import numpy as np


ARMS = {"left": slice(0, 7), "right": slice(7, 14)}
APPROACH_GAP = -0.009
PINCH = -0.0003
RELEASE = -0.015


class TapeDemo:
    def __init__(self, robot, video, limit):
        from yam_arena.constants import READY
        from yam_arena.station import BASES, TABLE_TOP

        self.robot = robot
        self.video = video
        self.limit = limit
        self.bases = BASES
        self.table = TABLE_TOP
        self.ready = np.r_[READY, 1.0]
        self.action = np.tile(self.ready, 2)
        self.steps = 0
        self.watched_panel = None
        self.phase = "settle"

    def rgb(self, camera):
        rgb = self.robot._base.scene[camera].data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        return rgb[0, ..., :3].cpu().numpy()

    def step(self):
        import cv2
        import torch

        if self.steps >= self.limit:
            raise RuntimeError("Expert reached its action limit")
        commands = torch.as_tensor(self.action, dtype=torch.float32, device=self.robot._base.device)
        _, _, terminated, truncated, _ = self.robot._env.step(commands[None, :])
        if (terminated | truncated).any():
            raise RuntimeError("Simulation reset during expert action")
        wrists = [cv2.resize(self.rgb(camera), (320, 180)) for camera in ("left_cam", "right_cam")]
        frame = np.vstack((self.rgb("top_cam"), np.hstack(wrists)))
        self.video.send(np.ascontiguousarray(frame))
        self.steps += 1
        if self.watched_panel is not None:
            nodes = self.robot.panel_nodes(self.watched_panel)
            if not np.isfinite(nodes).all() or np.abs(nodes).max() > 2.0:
                raise RuntimeError(f"Cloth physics became unstable during {self.phase} at step {self.steps}")

    def settle(self, count=20):
        for _ in range(count):
            self.step()

    def move(self, arm, waypoints, *, tilt=0.0):
        from yam_arena.constants import FINGER_OPEN
        from yam_arena.demo import ArmPlan, grip_rotation
        from yam_arena.ik import slerp_matrix, smoothstep
        from yam_arena.urdf import axis_angle_matrix

        class LeaningPlan(ArmPlan):
            def target(self, t):
                previous, rotation, finger = self.start_pos, self.start_rot, self.finger_start
                for waypoint in self.waypoints:
                    goal = np.asarray(waypoint.pos)
                    goal_rot = grip_rotation(waypoint.yaw) @ axis_angle_matrix(np.array([1, 0, 0]), tilt)
                    if t < waypoint.duration:
                        fraction = smoothstep(t / waypoint.duration)
                        return (previous + fraction * (goal - previous), slerp_matrix(rotation, goal_rot, fraction),
                                finger + fraction * (waypoint.finger - finger), waypoint.label)
                    t -= waypoint.duration
                    previous, rotation, finger = goal, goal_rot, waypoint.finger
                return previous, rotation, finger, "done"

        plan_type = LeaningPlan if tilt else ArmPlan
        joints = self.action[ARMS[arm]]
        plan = plan_type(waypoints, self.bases[arm], q_start=joints[:6], finger_start=joints[6] * FINGER_OPEN)
        for i in range(1, math.ceil(20 * plan.duration) + 1):
            action, self.phase = plan.action7(min(i / 20, plan.duration))
            self.action[ARMS[arm]] = action
            self.step()
        print(f"{arm} {waypoints[-1].label}: IK error {plan.worst_ik_mm:.1f} mm / {plan.worst_ik_deg:.1f} deg",
              flush=True)

    def park(self, arm):
        from yam_arena.ik import smoothstep

        joints = self.action[ARMS[arm]]
        start = joints.copy()
        for i in range(1, 31):
            joints[:] = start + smoothstep(i / 30) * (self.ready - start)
            self.step()

    def tab(self, index):
        from yam_arena.ik import quat_xyzw_to_matrix

        pose = self.robot._base.scene[f"tape_tab_{index}"].data.root_link_pose_w
        pose = pose.torch[0].cpu().numpy().copy()
        normal = quat_xyzw_to_matrix(pose[3:])[:, 1]
        yaw = (math.atan2(normal[0], -normal[1]) + math.pi / 2) % math.pi - math.pi / 2
        return pose[:3], yaw

    def grab(self, arm, index, clearance, lift, *, tilt=0.0, tcp_offset=0.006):
        from yam_arena.demo import Waypoint, grip_rotation
        from yam_arena.ik import tcp_pose
        from yam_arena.urdf import axis_angle_matrix

        tab, yaw = self.tab(index)
        rotation = grip_rotation(yaw) @ axis_angle_matrix(np.array([1, 0, 0]), tilt)
        pinch = tab - tcp_offset * rotation[:, 2]
        hover = (*tab[:2], self.table + clearance)
        check = pinch + (0, 0, lift)
        self.move(arm, [Waypoint(1.6, hover, APPROACH_GAP, "approach tab", yaw),
                        Waypoint(1.3, tuple(pinch), APPROACH_GAP, "straddle tab", yaw),
                        Waypoint(0.85, tuple(pinch), PINCH, "pinch tab", yaw)], tilt=tilt)
        robot_name = f"{arm}_robot"
        joints, fingers = self.robot._ids[robot_name]
        state = self.robot._base.scene[robot_name].data.joint_pos.torch[0].cpu().numpy()
        actual_tcp, _ = tcp_pose(state[joints], self.bases[arm])
        print(f"Pinch tracking offset (mm): {np.round(1000 * (actual_tcp - pinch), 1)}; "
              f"finger positions (mm): {np.round(1000 * state[fingers], 1)}", flush=True)
        self.move(arm, [Waypoint(1.0, tuple(check), PINCH, "lift tab", yaw)], tilt=tilt)
        actual, _ = self.tab(index)
        delta = actual[2] - tab[2]
        print(f"Panel {index} tape tab rose {delta * 1000:.1f} mm", flush=True)
        return check, yaw, delta >= lift * 0.45

    def release(self, arm, point, yaw, *, retract=0.05):
        from yam_arena.demo import Waypoint

        self.move(arm, [Waypoint(0.6, tuple(point), RELEASE, "release tab", yaw),
                        Waypoint(0.9, tuple(point + (0, 0, retract)), RELEASE, "withdraw", yaw)])
        self.park(arm)
        self.settle()

    def place_flat(self, arm, index, shift, *, lift=0.008, rotate=0.0, clearance=0.07):
        from yam_arena.demo import Waypoint

        point, yaw, held = self.grab(arm, index, clearance, lift)
        if held:
            goal = point + (*shift, 0)
            lower = goal - (0, 0, lift - 0.002)
            self.move(arm, [Waypoint(1.7, tuple(goal), PINCH, "shift held panel", yaw + rotate),
                            Waypoint(0.7, tuple(lower), PINCH, "lay panel down", yaw + rotate)])
            self.release(arm, lower, yaw + rotate)
        else:
            self.release(arm, point, yaw)
        return held

    def retrieve(self, layout):
        from layout import ZONE
        from yam_arena.demo import Waypoint

        index = layout.target_slot
        start, yaw, held = self.grab("right", index, clearance=0.125, lift=0.050, tilt=math.radians(30))
        if not held:
            self.release("right", start, yaw)
            return

        # Maintain the pinch angle while lifting through the stack and clearing the rim.
        self.watched_panel = index
        lean = math.radians(30)
        handoff = np.array((0.216, 0.051, self.table + 0.021))
        transfer = [
            (1.6, (start[0], start[1], self.table + 0.170), "lift vertically clear of the stack", lean),
            (2.3, (0.245, start[1], self.table + 0.190), "move above the middle of the bin", lean),
            (2.3, (0.213, -0.038, self.table + 0.165), "take cloth over the rim", lean),
            (1.4, (*handoff[:2], self.table + 0.105), "carry into shared workspace", lean),
            (1.5, tuple(handoff), "lay cloth down for left arm", 0.0),
        ]
        for seconds, point, label, tilt in transfer:
            self.move("right", [Waypoint(seconds, point, PINCH, label, yaw)], tilt=tilt)
            lowest = self.robot.panel_nodes(index)[:, 2].min()
            print(f"Lowest fabric point after {label}: {1000 * (lowest - self.table):.1f} mm above table",
                  flush=True)
        self.release("right", handoff, yaw)
        center = self.robot.panel_nodes(index)[:, :2].mean(0)
        print(f"Panel {index} after retrieval: center {center.round(3)}", flush=True)
        tab = self.robot.tab_position(index)
        if tab[1] > 0:
            self.place_flat("left", index, np.asarray(ZONE) - center, lift=0.06, clearance=0.09)

    def align(self):
        from layout import GOAL
        from scoring import grid

        center = self.robot.panel_nodes(0)[:, :2].mean(0)
        self.place_flat("right", 0, np.asarray(GOAL[0]) - center)
        brace, brace_yaw, _ = self.grab("right", 0, clearance=0.065, lift=0.0)
        first = grid(self.robot.panel_nodes(0))[-1, :, :2]
        second = grid(self.robot.panel_nodes(1))[0, :, :2]
        # Use the resting edges for alignment and account for drape during the 35 mm lift.
        shift = first.mean(0) - second.mean(0) - (0.0147, 0.0244)
        direction_first = first[-1] - first[0]
        direction_second = second[-1] - second[0]
        yaw_correction = math.atan2(direction_first[1], direction_first[0]) - math.atan2(
            direction_second[1], direction_second[0])
        self.place_flat("left", 1, shift, lift=0.035, rotate=yaw_correction)
        result = self.robot.measure()
        print(f"Seam geometry before release: {result.geometry_score:.2f}, {result.checks_passed} stations",
              flush=True)
        self.release("right", brace, brace_yaw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("02", "03"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import imageio_ffmpeg
    from inspect_robots import Scene
    from PIL import Image
    from task import MAX_STEPS, make_layout, make_robot

    args.output.mkdir(parents=True, exist_ok=True)
    layout = make_layout(0)
    robot = make_robot(layout, live=False, setup_image=None)
    status = 1
    try:
        robot.reset(Scene(id="layout-0", instruction=layout.instruction), seed=0)
        robot.verify_settled()
        video = imageio_ffmpeg.write_frames(
            str(args.output / "scripted-demo.mp4"), (640, 540), fps=20, codec="libx264",
            pix_fmt_out="yuv420p", macro_block_size=2, output_params=["-movflags", "+faststart", "-preset", "fast"],
        )
        try:
            video.send(None)
            demo = TapeDemo(robot, video, min(MAX_STEPS, 1200))
            demo.settle()
            Image.fromarray(demo.rgb("top_cam")).save(args.output / "setup.png")
            if args.task == "02":
                demo.retrieve(layout)
            else:
                demo.align()
            demo.settle(40)
            result = robot.measure()
            summary = {**asdict(result), "steps": demo.steps, "control_hz": 20, "seed": 0}
            (args.output / "scripted-score.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
            status = 0 if result.goal_met else 1
        finally:
            video.close()
    except Exception:
        traceback.print_exc()
    finally:
        if robot.app is not None:
            robot.app.close(exit_code=status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
