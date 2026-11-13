"""Score the simplified mechanism from relative joint travel and release."""

from dataclasses import dataclass

from .mechanism import END_TOLERANCE, LENGTH, PIN_SEATED_TOLERANCE, PIN_TRAVEL, PITCH, STARTED


@dataclass(frozen=True)
class Reading:
    slider_m: float
    pin_out_m: float
    released: bool


def zip_score(readings, *, initial_slider=STARTED):
    if any(reading.pin_out_m > PIN_SEATED_TOLERANCE for reading in readings):
        return 0.0
    final = readings[-1]
    remaining_travel = LENGTH - initial_slider
    progress = (final.slider_m - initial_slider) / remaining_travel
    if final.slider_m >= LENGTH - END_TOLERANCE and final.released:
        return 1.0
    return max(0.0, min(progress, 1.0 - PITCH / remaining_travel))


def unzip_score(readings):
    final = readings[-1]
    if final.slider_m <= END_TOLERANCE and final.pin_out_m >= PIN_TRAVEL - END_TOLERANCE and final.released:
        return 1.0
    opened = (LENGTH - final.slider_m) / LENGTH
    return max(0.0, min(opened, 1.0 - PITCH / LENGTH))
