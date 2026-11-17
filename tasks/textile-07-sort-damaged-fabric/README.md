# Sort damaged and intact fabric with tape tabs

> Put the cloth with the square hole fully inside the red zone labeled Damaged. Put the cloth without a hole fully inside the green zone labeled Undamaged. Use the raised tape tabs to pick up the cloths. Release both cloths inside their zones.

## Setup

Use two 140 × 100 × 5 mm teal-and-cream terry cloth panels. The panel at (170, −120) mm has a 25 × 25 mm hole. The panel at (170, 120) mm is intact. Center the 180 × 140 mm red Damaged zone at (380, 120) mm and the green Undamaged zone at (380, −120) mm. Use seed 0. Coordinates are measured in the tabletop plane with +x away from the arms and +y to robot-left. Fit both swatches with identical orange tabs on the top face, centered at an offset of (−44, 27) mm from each swatch center. For each tab, fold a 60 mm length of nominal 25 mm Pro Pocket Gaff cotton cloth tape over a 25 mm wide, 18 mm tall, 3.4 mm thick PETG stiffener. Press the two adhesive feet flat on the cloth. Each foot extends about 10 mm from the bottom of the stiffener. The exposed grip is 25 × 4 × 18 mm, width × thickness × height. [Cloth tape specifications](https://www.protapes.com/wp-content/uploads/2021/10/Pro-Pocket-Gaff.pdf)

## Scoring

Score half the fraction of each panel’s projected fabric area in its correct zone, summed over the panels. A held panel earns no credit. Exclude the hole and count overlapping folds once.

## Fidelity

Both 5 mm washcloths are surface deformable. The damaged swatch has a physical cutout. Each simulated 25 × 4 × 18 mm tape-wrapped PETG grip is a rigid block attached to the cloth mesh. The tape feet, peeling, and independent tape flex are omitted. Release uses 80 mm tool-center clearance from the swatch and its tab.

## Reproduce

Run from this directory on the RoboCurve YAM Isaac Lab Arena launchable. The first command previews the scene with parked arms. The second records a one-minute Dream-1 trial at 20 Hz. The third records the scripted demonstration.

```bash
source /data/IsaacLab-Arena/yam.env
python cloth_station_preview.py --seed 0 --headless
python cloth_dream_run.py --seed 0 --max-steps 1200 --log-dir runs/dream
python repair_sort_expert.py 07 --output runs/expert
```

## Physical task and evidence

[O*NET lists inspecting and rejecting nonconforming materials](https://www.onetonline.org/link/details/51-9061.00). [BLS reports about 602,000 U.S. jobs in the inspector, tester, sorter, sampler, and weigher group in 2025](https://www.bls.gov/ooh/production/quality-control-inspectors.htm). This scene represents the visual inspection and separation work of that role.

The recorded seed 0 Dream-1 run scored 0.00 in 1,200 actions at 20 Hz. The seed 0 scripted demo scored 1.00 in 952 actions at 20 Hz. The draft action limit defines the maximum for a trial.

For a physical trial, cut both panels from the same cotton terry cloth. Cut the hole before the run and keep the tabs identical on both swatches. Use matte red and green paper markers with the stated labels. Judge release from the video and area in each zone from a final overhead photo.
