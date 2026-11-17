# Retrieve matching fabric from an overlapping stack with tape tabs

> Retrieve the teal cotton panel with white stripes from the gray bin. Grip the raised tape tab on the matching panel and lift it clear of the bin. Move other panels aside if needed. Place the entire target panel inside the green STAGING zone and let go. Leave every other panel in the bin. If you take a different panel out, put it back.

## Setup

Five 140 × 100 mm cotton, linen, and felt swatches in a 240 × 180 × 80 mm gray bin centered at (280, −120) mm. Place the 180 × 140 mm green staging zone at (280, 150) mm. Use seed 0. From bottom to top, set rust plaid linen, plain yellow linen, purple dotted felt, teal striped cotton, and teal cotton with large white rings at offsets (−43, −32), (43, 32), (−43, 32), (43, −32), (6, 3) mm from the bin center. Coordinates are measured in the tabletop plane with +x away from the arms and +y to robot-left. Use nominal 1-inch (25 mm) orange Pro Pocket Gaff cotton cloth tape. Fold the tape over a 25 mm wide, 18 mm tall, 3.4 mm thick PETG stiffener. Bend the two adhesive ends outward at the bottom and press them onto the patterned face as mounting feet. The exposed grip is 25 mm wide, about 4 mm thick with two tape layers, and 18 mm tall. Keep the gripper faces free of exposed adhesive. Each swatch has one identical raised tab. For the five layers bottom to top, place the tab center at these offsets from its own swatch center, in mm: (−39, −24), (54, 16), (−44, 23), (54, −20), (2, 8).

A rigid open bin holds five overlapping, initially flat deformable cloth layers. The stack approximates a mixed fabric bin. Fabric thickness ranges from 3.3 to 4.8 mm. Release is estimated from gripper clearance. The raised tape grip is modeled as a light rigid block shaped like the tape-wrapped stiffener attached to a small patch of the deformable mesh. The simulated tape neither peels nor flexes independently.

The real-world tape reference is [Pro Pocket Gaff, 25 mm wide and 0.28 mm thick](https://www.protapes.com/wp-content/uploads/2021/10/Pro-Pocket-Gaff.pdf). Cut a strip about 60 mm long so each face covers the stiffener and each mounting foot has about 10 mm of contact with the cloth.

The tape tabs move with the cloth through physics attachments. The scorer measures the fabric panels, and its release check includes the tab positions.

## Review and run

The saved layout preview uses seed 0 and shows the initial geometry before physics settling. This variant has its own evidence. Run from this task directory.

```bash
source /data/IsaacLab-Arena/yam.env
python task.py --seed 0 --setup-image runs/tabbed-preview/setup.png
```

To repeat the recorded trial, run Dream-1 for one simulated minute at 20 Hz.

```bash
python task.py --seed 0 --dream --max-steps 1200 \
  --setup-image runs/tabbed-dream/setup.png --log-dir runs/tabbed-dream
```


```bash
python tabbed_expert.py 02 --output runs/tabbed-expert
```

## Physical task and evidence

The physical challenge is extracting the tucked swatch past both overlapping material and the bin wall. The temporary tape-wrapped grips are specified in the setup for physical replication. The grips have the same construction and color across all five fabrics.

The recorded seed 0 Dream-1 run scored 0.00 in 1,200 actions at 20 Hz. The seed 0 scripted demo scored 1.00 in 640 actions at 20 Hz. The draft action limit defines the maximum for a trial.

For a physical trial, use a gray plastic sewing organizer or open bin with the stated interior layout. Measure the visible cloth footprints from an overhead photo after the gripper opens. Check the release and that the neighboring scraps remain inside. The simulator reports the continuous geometry score and a separate completion flag that requires release.
