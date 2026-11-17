"""Demonstrate the pickup with the seeded matching bottle as expert input."""

from dataclasses import asdict
import json
import math

import numpy as np

from layout import CONTROL_HZ, SHELF_SIZE, SLOTS, ZONE_CENTER


def plan_for(layout, table):
    from yam_arena.demo import Waypoint

    x, y = SLOTS[layout.target_slot]
    side = "left" if y > 0 else "right"
    yaw = 0.0
    travel = table + 0.150
    pickup = table + SHELF_SIZE[2] + 0.058
    landing = table + 0.060
    open_grip = -0.030
    closed_grip = -0.001
    zx, zy = ZONE_CENTER

    return side, [
        Waypoint(1.7, (x, y, travel), open_grip, "approach matching bottle", yaw),
        Waypoint(1.45, (x, y, pickup), open_grip, "straddle bottle", yaw),
        Waypoint(0.85, (x, y, pickup), closed_grip, "pinch body", yaw),
        Waypoint(1.0, (x, y, travel), closed_grip, "lift clear of shelf", yaw),
        Waypoint(1.4, (zx, y, travel), closed_grip, "withdraw from the row", yaw),
        Waypoint(1.6, (zx, zy, travel), closed_grip, "carry over the staging zone", yaw),
        Waypoint(1.4, (zx, zy, landing), closed_grip, "set bottle on table", yaw),
        Waypoint(0.75, (zx, zy, landing), open_grip, "release bottle", yaw),
        Waypoint(1.0, (zx, zy, travel), open_grip, "retreat", yaw),
    ]


def record_expert(robot, layout, output, max_steps):
    import cv2
    import imageio_ffmpeg
    from PIL import Image
    from yam_arena.constants import FINGER_OPEN, READY
    from yam_arena.demo import ArmPlan
    from yam_arena.ik import smoothstep
    from yam_arena.station import BASES, TABLE_TOP

    side, waypoints = plan_for(layout, TABLE_TOP)
    indices = slice(0, 7) if side == "left" else slice(7, 14)
    ready = np.r_[READY, 1.0]
    commands = np.tile(ready, 2)
    plan = ArmPlan(waypoints, BASES[side], q_start=READY, finger_start=FINGER_OPEN)
    motion_steps = math.ceil(plan.duration * CONTROL_HZ)
    if motion_steps + 90 > max_steps:
        raise ValueError(f"The scripted motion needs {motion_steps + 90} steps")
    video = imageio_ffmpeg.write_frames(
        str(output / "scripted-demo.mp4"), (640, 540), fps=CONTROL_HZ, codec="libx264",
        pix_fmt_out="yuv420p", macro_block_size=2, output_params=["-movflags", "+faststart", "-preset", "fast"],
    )
    steps = 0

    def step():
        nonlocal steps
        robot.command(commands)
        wrists = [cv2.resize(robot.frame(name), (320, 180)) for name in ("left_cam", "right_cam")]
        frame = np.vstack((robot.frame("top_cam"), np.hstack(wrists)))
        video.send(np.ascontiguousarray(frame))
        steps += 1

    try:
        video.send(None)
        Image.fromarray(robot.frame("top_cam")).save(output / "start.png")
        for _ in range(20):
            step()
        phase = None
        for i in range(1, motion_steps + 1):
            commands[indices], label = plan.action7(min(plan.duration, i / CONTROL_HZ))
            if label != phase:
                print(f"{side}: {label}", flush=True)
                phase = label
            step()
        previous = commands[indices].copy()
        for i in range(1, 31):
            commands[indices] = previous + smoothstep(i / 30) * (ready - previous)
            step()
        for _ in range(40):
            step()
        print(f"Maximum IK error {plan.worst_ik_mm:.2f} mm / {plan.worst_ik_deg:.2f} degrees", flush=True)
        result = robot.measure()
        report = {
            **asdict(result), "seed": layout.seed, "steps": steps, "control_hz": CONTROL_HZ,
            "target_slot": layout.target_slot, "target": f"{layout.target.name} {layout.target.strength}",
        }
        Image.fromarray(robot.frame("top_cam")).save(output / "final.png")
        (output / "scripted-score.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return 0 if result.goal_met else 1
    finally:
        video.close()
