"""TextileBench Task 13. Close an already started separating zipper."""

from zipper.mechanism import STARTED
from zipper.runner import run
from zipper.scoring import zip_score

TASK = "textile-13-zip-started"
INSTRUCTION = (
    "Use your left arm to pin the jacket down against the table so it stays still. "
    "While holding it down, use your right arm to grab the orange zipper pull and zip the jacket up to the collar. "
    "If it stops short, regrasp the orange pull and finish sliding it to the top stop. "
    "Let go once the zipper reaches the top."
)

if __name__ == "__main__":
    raise SystemExit(run(TASK, INSTRUCTION, STARTED, zip_score, allow_open_start=True, auxiliary_grips=False))
