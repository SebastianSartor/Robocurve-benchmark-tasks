"""Scripted two-arm expert for the open-start padded jacket."""

import argparse
import json
import math
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .jacket_fabric import FABRIC_NAME, THICKNESS
from .mechanism import CONTROL_HZ, END_TOLERANCE, LENGTH, PIN_SEATED_TOLERANCE, STARTED
from .scoring import zip_score


ARM_ACTION = {"left": slice(0, 7), "right": slice(7, 14)}


class ZipExpert:
    def __init__(self, robot, video):
        from yam_arena.constants import READY

        self.robot = robot
        self.ready = np.r_[READY, 1.0]
        self.action = np.tile(self.ready, 2)
        self.video = video
        self.history = [robot.reading()]

    def step(self):
        command = self.robot._torch.as_tensor(
            self.action, dtype=self.robot._torch.float32, device=self.robot._base.device,
        )
        self.robot._env.step(command.unsqueeze(0))
        self.history.append(self.robot.reading())
        if self.video is not None:
            self.video.send(self.frame())

    def frame(self):
        import cv2

        def rgb(name):
            buffer = self.robot._base.scene[name].data.output["rgb"]
            buffer = buffer.torch if hasattr(buffer, "torch") else buffer
            return buffer[0, ..., :3].cpu().numpy()

        wrists = [cv2.resize(rgb(name), (320, 180)) for name in ("left_cam", "right_cam")]
        return np.ascontiguousarray(np.vstack((rgb("top_cam"), np.hstack(wrists))))

    def move(self, side, waypoints):
        from yam_arena.constants import FINGER_OPEN
        from yam_arena.demo import ArmPlan
        from yam_arena.station import BASES

        target = ARM_ACTION[side]
        current = self.action[target]
        plan = ArmPlan(waypoints, BASES[side], q_start=current[:6], finger_start=current[6] * FINGER_OPEN)
        for i in range(1, math.ceil(plan.duration * CONTROL_HZ) + 1):
            self.action[target] = plan.action7(i / CONTROL_HZ)[0]
            self.step()
        print(f"{side} {waypoints[-1].label}: IK {plan.worst_ik_mm:.1f} mm, {plan.worst_ik_deg:.1f} deg; "
              f"zipper {self.history[-1]}", flush=True)

    def hand_pose(self, side):
        from yam_arena.ik import tcp_pose
        from yam_arena.station import BASES

        name = f"{side}_robot"
        joint_ids, finger_ids = self.robot._ids[name]
        measured = self.robot._base.scene[name].data.joint_pos.torch[0].cpu().numpy()
        tip, _ = tcp_pose(measured[joint_ids], BASES[side])
        return tip, -sum(measured[finger_ids])

    def grasp_tab(self, yaw):
        from yam_arena.demo import Waypoint

        x, y, center_z = self.robot.handle_positions()[1]
        z = center_z + 0.004
        opened, pinched = -0.018, -0.004
        self.move("right", [
            Waypoint(1.5, (x, y, z + 0.075), opened, "approach", yaw),
            Waypoint(1.3, (x, y, z), opened, "descend", yaw),
            Waypoint(0.8, (x, y, z), pinched, "grasp", yaw),
        ])
        tip, jaw_gap = self.hand_pose("right")
        delta = (tip - self.robot.handle_positions()[1]) * 1000
        print(f"right measured TCP minus handle (mm): {np.round(delta, 1)}; jaw gap {jaw_gap * 1000:.1f} mm",
              flush=True)

    def brace_sleeve(self):
        from yam_arena.constants import PAD_TIP_PAST_TCP
        from yam_arena.demo import Waypoint

        nodes = self.robot._base.scene[FABRIC_NAME].data.nodal_pos_w.torch[0].cpu().numpy()
        x, y = 0.252, 0.133
        distance = np.linalg.norm(nodes[:, :2] - (x, y), axis=1)
        nearest = np.argpartition(distance, 5)[:5]
        z = float(np.median(nodes[nearest, 2])) + THICKNESS / 2 + PAD_TIP_PAST_TCP - 0.004
        self.move("left", [
            Waypoint(1.5, (x, y, z + 0.065), -0.004, "approach sleeve"),
            Waypoint(1.3, (x, y, z), -0.004, "brace sleeve"),
        ])
        return (x, y, z)

    def release(self, side, point, *, yaw=0.0):
        from yam_arena.demo import Waypoint
        from yam_arena.ik import smoothstep

        x, y, z = point
        self.move(side, [
            Waypoint(0.65, (x, y, z), -0.018, "open", yaw),
            Waypoint(1.1, (x, y, z + 0.075), -0.022, "retreat", yaw),
        ])
        target = ARM_ACTION[side]
        start = self.action[target].copy()
        for i in range(1, 31):
            self.action[target] = start + smoothstep(i / 30) * (self.ready - start)
            self.step()

    def run(self):
        from yam_arena.demo import Waypoint
        from yam_arena.urdf import quat_xyzw_to_matrix

        pull_yaw = math.pi / 2
        self.grasp_tab(pull_yaw)
        brace = self.brace_sleeve()
        for attempt in range(3):
            tab = self.robot.handle_positions()[1] + (0, 0, 0.004)
            rotation = self.robot.jacket.data.root_link_pose_w.torch[0, 3:].cpu().numpy()
            direction = quat_xyzw_to_matrix(rotation)[:, 0]
            travel = max(0.0, LENGTH - self.history[-1].slider_m)
            finish = tab + direction * (travel + 0.008)
            self.move("right", [
                Waypoint(max(1.2, travel / 0.041), tuple(finish), -0.004, "pull to top", pull_yaw),
                Waypoint(0.45, tuple(finish), -0.004, "settle at top", pull_yaw),
            ])
            hand, _ = self.hand_pose("right")
            delta = (hand - self.robot.handle_positions()[1]) * 1000
            print(f"end of pull {attempt + 1}: TCP minus tab (mm) {np.round(delta, 1)}", flush=True)
            if self.history[-1].slider_m >= LENGTH - END_TOLERANCE or attempt == 2:
                self.release("right", hand, yaw=pull_yaw)
                break
            x, y, z = hand
            self.move("right", [
                Waypoint(0.6, (x, y, z), -0.018, "open for regrasp", pull_yaw),
                Waypoint(1.0, (x, y, z + 0.055), -0.018, "clear tab", pull_yaw),
            ])
            self.grasp_tab(pull_yaw)
        self.release("left", brace)
        for _ in range(20):
            self.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Show the expert in the Brev desktop.")
    parser.add_argument("--hold-open", type=int, default=0, help="Seconds to show the final view after a live run.")
    parser.add_argument("--output", type=Path, help="Save a video and score to a new directory.")
    parser.add_argument("--benchmark-start", action="store_true", help="Begin with 20 mm already zipped.")
    args = parser.parse_args()
    if args.hold_open < 0 or (args.hold_open and not args.live):
        parser.error("--hold-open requires --live and a nonnegative number of seconds")

    from inspect_robots import Scene

    from .yam import ZipperEmbodiment

    instruction = "Brace the sleeve, close the orange zipper, and release the jacket."
    initial_slider = STARTED if args.benchmark_start else 0.0
    robot = ZipperEmbodiment("textile-13-zip-expert", instruction, initial_slider, live=args.live, auxiliary_grips=False)
    status = 1
    try:
        with ExitStack() as stack:
            video = None
            if args.output is not None:
                import imageio_ffmpeg

                args.output.mkdir(parents=True)
                video = imageio_ffmpeg.write_frames(
                    str(args.output / "zipper-expert.mp4"), (640, 540), fps=CONTROL_HZ, codec="libx264",
                    macro_block_size=2, output_params=["-movflags", "+faststart", "-preset", "fast"],
                )
                stack.callback(video.close)
                video.send(None)
            robot.reset(Scene(id="zip-started" if args.benchmark_start else "zip-open", instruction=instruction), seed=0)
            expert = ZipExpert(robot, video)
            initial = expert.history[0]
            if abs(initial.slider_m - initial_slider) > END_TOLERANCE or abs(initial.pin_out_m) > PIN_SEATED_TOLERANCE:
                raise RuntimeError(f"Unexpected initial slider or pin position: {initial}")
            expert.run()
            result = {
                "score": zip_score(expert.history, initial_slider=initial_slider),
                "initial": asdict(initial),
                "final": asdict(expert.history[-1]),
                "steps": len(expert.history) - 1,
                "control_hz": CONTROL_HZ,
            }
            print(json.dumps(result, indent=2), flush=True)
            if args.output is not None:
                (args.output / "score.json").write_text(json.dumps(result, indent=2) + "\n")
            status = 0 if result["score"] == 1.0 else 1
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
