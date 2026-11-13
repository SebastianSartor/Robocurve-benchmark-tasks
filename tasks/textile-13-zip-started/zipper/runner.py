"""Preview or evaluate a zipper pilot in the RoboCurve Brev environment."""

import argparse
import traceback
from functools import partial
from pathlib import Path

from .mechanism import MAX_STEPS
from .scoring import Reading


def run(task_name, instruction, initial_slider, score, *, allow_open_start=False, auxiliary_grips=True):
    parser = argparse.ArgumentParser(description=instruction)
    parser.add_argument("--dream", action="store_true", help="Use Dream-1. Otherwise check the starting scene.")
    parser.add_argument("--live", action="store_true", help="Show Isaac Sim on the Brev desktop.")
    parser.add_argument("--max-steps", type=int, default=200, help=f"Dream control steps (up to {MAX_STEPS}).")
    parser.add_argument("--hold-open", type=int, default=0, help="Seconds to leave the live viewport open at the end.")
    parser.add_argument("--log-dir", type=Path, help="Required for Dream. Use a new directory per run.")
    if allow_open_start:
        parser.add_argument("--start-open", action="store_true", help="Start with the slider at the hem and the pin seated.")
    parser.set_defaults(start_open=False)
    args = parser.parse_args()
    if args.dream and args.log_dir is None:
        parser.error("--dream requires --log-dir")
    if not 1 <= args.max_steps <= MAX_STEPS:
        parser.error(f"--max-steps must be between 1 and {MAX_STEPS}")
    if args.hold_open < 0 or (args.hold_open and not args.live):
        parser.error("--hold-open must be nonnegative and requires --live")
    if args.start_open:
        initial_slider = 0.0
        score = partial(score, initial_slider=initial_slider)
        task_name += "-open-start"
        instruction = "The orange zipper pull starts at the hem, with the pin already seated. " + instruction

    from inspect_robots import Scene, Task, eval
    from inspect_robots.scorer import Score

    from .yam import ZipperEmbodiment

    class ZipperScorer:
        name = "zipper_progress"

        def __call__(self, record, target):
            readings = [Reading(**step.result.info["zipper"]) for step in record.steps]
            if not readings:
                return Score(value=None, explanation="No recorded zipper state")
            final = readings[-1]
            return Score(
                value=score(readings),
                metadata={"slider_m": final.slider_m, "pin_out_m": final.pin_out_m, "released": final.released},
            )

    scene = Scene(id="flat-jacket", instruction=instruction)
    robot = ZipperEmbodiment(task_name, instruction, initial_slider, live=args.live, auxiliary_grips=auxiliary_grips)
    exit_code = 1
    try:
        if args.dream:
            task = Task(name=task_name, scenes=[scene], scorer=ZipperScorer(), max_steps=args.max_steps)
            print(f"Dream-1 instruction: {instruction}", flush=True)
            [log] = eval(
                task, "dream", robot, seed=0, log_dir=str(args.log_dir),
                fail_on_error=True, guard=["clamp", "delta-limit"], store_frames=True,
            )
            for sample in log.samples:
                print(f"{sample.scene_id}: {sample.status}; {sample.reduced}", flush=True)
            if log.status == "success":
                exit_code = 0
                if args.hold_open:
                    robot.hold_view(args.hold_open)
        else:
            robot.preview(scene, hold_open=args.hold_open)
            exit_code = 0
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
    finally:
        if robot.app is not None:
            robot.app.close(exit_code=exit_code)
    return exit_code
