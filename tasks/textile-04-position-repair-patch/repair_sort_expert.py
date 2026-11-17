"""Film scored, physical tape-tab demonstrations for TextileBench 04 and 07."""

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import traceback

import numpy as np
from tabbed_expert import PINCH, TapeDemo


SORT_FULL_SCORE = 0.99938 - 0.000005


class RepairSortDemo(TapeDemo):
    def repair_patch(self):
        from layout import BASE_THICKNESS
        from mesh import HOLE_LOOP
        from yam_arena.demo import Waypoint

        arm, patch = "left", 1
        lift = 0.026
        self.watched_panel = patch
        for _ in range(3):
            patch_center = self.robot.panel_nodes(patch)[:, :2].mean(0)
            hole_center = self.robot.panel_nodes(0)[HOLE_LOOP, :2].mean(0)
            point, yaw, held = self.grab(arm, patch, clearance=0.067, lift=lift, tcp_offset=0.007)
            if not held:
                self.release(arm, point, yaw)
                continue

            top = point + (*(hole_center - patch_center), 0)
            landing = top - (0, 0, lift - BASE_THICKNESS - 0.001)
            self.move(arm, [Waypoint(1.6, tuple(top), PINCH, "align patch above hole", yaw),
                            Waypoint(1.25, tuple(landing), PINCH, "lower patch flat", yaw)])
            for _ in range(3):
                result = self.robot.measure()
                delta = (np.asarray(result.hole_center_xy_m)
                         - self.robot.panel_nodes(patch)[:, :2].mean(0))
                print(f"Patch coverage {result.hole_coverage:.3f}, margin {result.margin_square_coverage:.3f}, "
                      f"center error {1000 * delta} mm", flush=True)
                if result.margin_square_coverage >= 0.99999 or np.linalg.norm(delta) < 0.0006:
                    break
                landing = landing + (*np.clip(delta, -0.004, 0.004), 0)
                self.move(arm, [Waypoint(0.65, tuple(landing), PINCH, "center patch", yaw)])
            self.release(arm, landing, yaw)
            reading = self.robot.measure()
            print(f"Released patch: {reading}", flush=True)
            if reading.goal_met:
                return

    def move_swatch(self, arm, slot, destination):
        self.watched_panel = slot
        current = self.robot.panel_nodes(slot)[:, :2].mean(0)
        return self.place_flat(arm, slot, np.asarray(destination) - current, lift=0.058, clearance=0.085)

    def sort_swatches(self):
        from cloth_station_preview import ZONES

        shared_table = (0.235, 0.0)
        for label, slot in (("Damaged", self.robot.damaged_slot), ("Undamaged", 1 - self.robot.damaged_slot)):
            destination = np.asarray(ZONES[label][0])
            source = self.robot.panel_nodes(slot)[:, :2].mean(0)
            donor = "left" if source[1] > 0 else "right"
            receiver = "left" if destination[1] > 0 else "right"
            if donor != receiver and not self.move_swatch(donor, slot, shared_table):
                print(f"The {label} swatch was not transferred to the shared workspace.", flush=True)
                continue
            print(f"{label} swatch handoff center: {self.robot.panel_nodes(slot)[:, :2].mean(0)}", flush=True)
            self.move_swatch(receiver, slot, destination)
            result = self.robot.measure()
            print(f"{label} after placement: {result['panels'][label]}", flush=True)
            if result['panels'][label]['fraction_in_zone'] < 0.999:
                self.move_swatch(receiver, slot, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("04", "07"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import imageio_ffmpeg
    from inspect_robots import Scene
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=True)
    if args.task == "04":
        from layout import make_layout
        from task import make_robot

        layout = make_layout(0)
        instruction = layout.instruction
        robot = make_robot(layout, live=False, setup_image=None)
    else:
        from cloth_dream_run import ClothStationEmbodiment
        from cloth_station_preview import INSTRUCTION as instruction

        robot = ClothStationEmbodiment(0)
    status = 1
    try:
        robot.reset(Scene(id="layout-0", instruction=instruction), seed=0)
        robot.verify_settled()
        video = imageio_ffmpeg.write_frames(
            str(args.output / "scripted-demo.mp4"), (640, 540), fps=20, codec="libx264",
            pix_fmt_out="yuv420p", macro_block_size=2, output_params=["-movflags", "+faststart", "-preset", "fast"],
        )
        try:
            video.send(None)
            demo = RepairSortDemo(robot, video, 1800)
            demo.settle()
            Image.fromarray(demo.rgb("top_cam")).save(args.output / "setup.png")
            if args.task == "04":
                demo.repair_patch()
            else:
                demo.sort_swatches()
            demo.settle(40)
            measured = robot.measure()
            result = asdict(measured) if is_dataclass(measured) else {
                **measured, "goal_met": measured["score"] >= SORT_FULL_SCORE,
            }
            report = {**result, "steps": demo.steps, "control_hz": 20, "seed": 0}
            (args.output / "scripted-score.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            status = 0 if result["goal_met"] else 1
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
