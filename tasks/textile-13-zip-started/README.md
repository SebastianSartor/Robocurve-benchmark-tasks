# Zip an already-started baby jacket

Use your left arm to pin the jacket down against the table so it stays still. While holding it down, use your right arm to grab the orange zipper pull and zip the jacket up to the collar. If it stops short, regrasp the orange pull and finish sliding it to the top stop. Let go once the zipper reaches the top.

Closing a garment zipper requires holding flexible fabric steady while guiding the pull along a narrow path.

## Reproduce

Run from this folder on the RoboCurve YAM Isaac Lab Arena launchable. The fixed evidence layout is seed 0. The maximum runtime is 240 simulated seconds at 20 Hz.

```bash
source /data/IsaacLab-Arena/yam.env
python -m textile_13_zip_started --dream --max-steps 1200 --log-dir runs/dream
python -m zipper.expert --benchmark-start --output runs/demo
```

The recorded Dream run used 1,200 steps (one minute). The Dream command records the three robot cameras and the evaluation JSON. The scripted command records the robot cameras and writes a measured score JSON. Use a fresh run directory for a new attempt.

## Setup

Lay a padded baby jacket front up and hem toward the arms. The cloth is about 240 mm long and 370 mm across the extended sleeves, with an 8 mm fabric thickness. The zipper is 180 mm long. Start at 20 mm above the hem with the pin seated and the orange pull exposed. The zipper origin is at (145, 0) mm. Coordinates are measured in the tabletop plane with +x away from the arms and +y to robot-left.

## Scoring

Score the fraction of the remaining 160 mm of travel closed. Score zero if the pin protrudes more than 3 mm at any point. Award 1.0 if the slider is within 3 mm of the top and both gripper tool centers are more than 40 mm from the garment and zipper. Otherwise cap the score at 0.9625.

## Fidelity and assets

The continuous deformable jacket has a sewn hem bridge. Rigid internal rails approximate zipper travel and the moving teeth are rendered from slider position. Individual teeth and cloth snags are outside the physics model.

Task-specific fabric meshes and textures are generated from the included source. The robot, calibrated station and controller come from the RoboCurve YAM Isaac Lab Arena launchable. Dependencies include NumPy, Pillow, Shapely 2.x, Trimesh, Inspect Robots and the launchable's Isaac Lab packages.

Recorded environment: Isaac Lab `37ddf6268717`, Isaac Lab Arena `c7edf3a9d5d8`, and YAM Arena `0059d02a1a91`.
