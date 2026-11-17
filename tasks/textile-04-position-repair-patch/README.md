# Position a repair patch over a fabric hole with tape tabs

> Grip the raised tape tab on the loose green square patch and place the patch over the square hole in the blue cloth. Cover the entire opening, with at least 2 millimeters of patch extending past each side. Smooth the patch flat against the blue cloth and let go.

## Setup

Lay a blue 140 × 100 × 3.2 mm cloth centered at (170, 0) mm. It has a centered 25 × 25 mm square cutout. Lay a green 35 × 35 × 2.4 mm cloth patch centered at (170, 91) mm, entirely separate from the base. Both lie flat on the work surface. Coordinates are measured in the tabletop plane with +x away from the arms and +y to robot-left. Cut a 48 mm length of nominal 25 mm orange Pro Pocket Gaff cotton cloth tape and trim it to 13 mm wide. Fold it over a 13 mm wide, 15 mm tall, 3.4 mm thick PETG stiffener. Press the two adhesive feet flat on the green patch's top face, centered at an offset of (−8, 8) mm from the patch center. Each foot extends about 7 mm from the bottom of the stiffener. The raised grip is 13 × 4 × 15 mm, width × thickness × height. [Cloth tape specifications](https://www.protapes.com/wp-content/uploads/2021/10/Pro-Pocket-Gaff.pdf)

## Scoring

After release, score 0.6 times the fraction of the opening covered plus 0.4 times the fraction of the centered 29 × 29 mm margin square covered, viewed from above. For goal completion the patch must rest on the base. Folded-over patch area counts once.

## Fidelity

The two pieces are deformable. The base has a mesh cutout and patch coverage is measured on cloth geometry. The real raised grip is trimmed cotton cloth tape folded over a PETG stiffener. In simulation it is a solid 13 × 4 × 15 mm block attached to a small region of the patch. The tape feet, peeling, and independent tape flex are omitted. Release is estimated from gripper clearance to the cloth and the tab.

## Reproduce

Run from this directory on the RoboCurve YAM Isaac Lab Arena launchable. The first command previews the scene with parked arms. The second records a one-minute Dream-1 trial at 20 Hz. The third records the scripted demonstration.

```bash
source /data/IsaacLab-Arena/yam.env
python task.py --seed 0 --setup-image runs/preview/setup.png
python task.py --seed 0 --dream --max-steps 1200 --log-dir runs/dream
python repair_sort_expert.py 04 --output runs/expert
```

## Physical task and evidence

The 25 mm opening and 2 mm margin make the endpoint directly measurable from an overhead image. Place a millimeter scale beside the work area for a physical evaluation. The trial ends with the patch laid in place, before stitching or bonding.

The recorded seed 0 Dream-1 run scored 0.00 in 1,200 actions at 20 Hz. The seed 0 scripted demo scored 1.00 in 292 actions at 20 Hz. The draft action limit defines the maximum for a trial.

For a physical trial, use blue heavy cotton canvas for the base and a green cotton twill patch. Mark the required margin with a 29 mm square reference on a transparent ruler or overhead photo. Check that the placed patch rests flat after the gripper opens.
