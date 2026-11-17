# PharmacyBench 01 · Pick the right stock bottle

> Read the name and strength on the prescription card. Pick the stock bottle that matches both. Place it upright, completely inside the green STAGING zone, and let go. Leave every other bottle in its marked shelf slot.

The shelf has six physically identical bottles. NERAVEX and NERAVIX each come in 10 mg and 25 mg in this fictional inventory. Every prescription has a name look-alike at the requested strength and an exact name at the wrong strength. The other two bottles have distinct names and the same appearance. A separate prescription card gives the request, and labels are printed both on the front and cap to support wrist and overhead inspection.

[Task metadata](task.json) gives the measured setup and scoring rubric.

## Reproduce

Run from this folder on the existing RoboCurve YAM Isaac Lab Arena launchable. Seed 0 requests NERAVIX 25 mg. Change the seed to vary both the prescription and shelf order. See the [sample label textures](assets/) for the seed 0 artwork, or let the runner generate textures for any seed. It needs the standard launchable dependencies plus Pillow, NumPy, Shapely and either DejaVu Sans Bold or Arial Bold.

```bash
source /data/IsaacLab-Arena/yam.env

# Inspect the initial arrangement on a Mac or Brev.
python task.py --layout-only --seed 0

# Render a settled scene, including close-ups of the labels and prescription.
python task.py --seed 0 --output runs/preview-seed0

# Record a physical pickup using the published seed as the expert input.
python task.py --seed 0 --expert --output runs/expert-seed0

# Evaluate Dream-1 with the 120-second benchmark limit.
python task.py --seed 0 --dream --max-steps 2400 --output runs/dream-seed0
```

Pass `--live` to show the expert or Dream trial in the Brev desktop. The scripted expert uses the layout's matching stock index. Dream receives the same generic instruction on every seed and locates the match from the camera views.

The setup mirrors the PharmacyBench document's six bottles, seeded layout and scoring. This version uses independently authored bottle geometry and artwork so each bottle can carry a precise, readable label with shared collision geometry. A comparable reference is the white packer bottle in [NVIDIA RoboLab's object inventory](https://github.com/NVlabs/RoboLab/blob/main/assets/objects/README.md).

## Physical task and evidence

[O*NET lists prescription preparation as a core pharmacy technician duty](https://www.onetonline.org/link/details/29-2052.00) and [prescription verification and dispensing for pharmacists](https://www.onetonline.org/link/details/29-1051.00). [BLS reports about 471,200 technicians and 325,200 pharmacists in 2025](https://www.bls.gov/emp/tables/occupational-projections-and-characteristics.htm). These are occupation totals for the dispensing workflow. The expert receives the seeded matching stock index to demonstrate physical handling. Dream receives the same generic prompt for every seed. [FDA background on look-alike names and medication errors](https://www.fda.gov/drugs/medication-errors-related-cder-regulated-drug-products/fda-name-differentiation-project).

The recorded seed 0 Dream-1 run scored 0.00 in 1,200 actions at 20 Hz. The seed 0 scripted demo scored 1.00 in 313 actions at 20 Hz. The draft action limit defines the maximum for a trial.

For a physical trial, load identical opaque packer bottles with inert ballast to the stated mass and attach the printed labels and card. Measure the 70 mm center spacing and the staging outline with a ruler. Check upright placement and which bases remain in their slots from the final view.
