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

Lay a padded baby jacket front up, with its sleeves extended and its hem toward the arms. For a physical replica, use a woven polyester shell and lining with quilted polyester batting. Match the simulated dimensions of approximately 240 mm hem to collar, 370 mm cuff to cuff, and 8 mm uncompressed fabric thickness.

Use a one-way, open-end separating garment zipper with 180 mm of slider travel, molded acetal teeth on woven polyester tape, and a zinc-alloy non-locking slider. Use the zipper’s bottom insertion pin and retaining box. A [YKK VISLON Standard open-end zipper](https://www.ykkfastening.com/products/search/detail.html?pdid1=VISLON%28R%29+Standard) is a reference for the molded-tooth design. Select a zipper and matching slider that meet the specified materials and travel.

### Physical pull attachment

1. 3D-print a rectangular block in orange PETG at a **0.2 mm layer height** and **40% infill**, then measure it with calipers. Make it **18 mm along the zipper, 16 mm across the zipper, and 24 mm tall**. Keep both opposing grip faces flat.
2. Remove the stock hinged tab. Dry-fit the block directly over the slider crown. If needed, fit a rigid PETG spacer below the block to put its center about **36 mm above the tabletop** when the jacket lies flat. Keep the spacer below the gripping faces.
3. Degrease the mating faces, lightly roughen them with sandpaper, and remove the dust. Mix equal parts of **J-B Weld PlasticBonder Tan (SKU 50133)**, a two-part urethane adhesive. Spread a thin layer on the mating faces, center the block, and hold it in place with light clamping pressure. Wipe away squeeze-out before it sets. Keep the adhesive clear of the slider channel, teeth, and tape. Let it cure for at least **one hour at room temperature** and follow any longer cure time printed on the package. Once cured, tug the block by hand to check for movement, then run the slider through its full travel to verify that it moves freely. [Adhesive instructions](https://www.jbweld.com/product/plastic-bonder-syringe)
4. Seat the insertion pin fully, move the slider **20 mm up from the hem**, and leave the orange block exposed. Place the zipper origin at **(145, 0) mm** in the tabletop plane, with +x away from the arms and +y to robot-left.

## Scoring

Score the fraction of the remaining 160 mm of travel closed. Score zero if the pin protrudes more than 3 mm at any point. Award 1.0 if the slider is within 3 mm of the top and both gripper tool centers are more than 40 mm from the garment and zipper. Otherwise cap the score at 0.9625.

## Fidelity and assets

The simulation represents the jacket shell, lining, and batting as one continuous deformable surface with a sewn hem bridge. The orange pull is an oversized solid **18 × 16 × 24 mm** cuboid rigidly fixed to the slider. The physical reference is the custom 3D-printed PETG attachment with 40% infill described above. Its wider, thicker grasp faces and fixed orientation make it easier to grasp than a typical thin, hinged tab. Rigid internal rails approximate zipper travel. The visible teeth follow slider position, and the physics omits individual tooth meshing and cloth snags. The physical replica is specified here for reproduction. The recorded trials used the simulator.

Task-specific fabric meshes and textures are generated from the included source. The robot, calibrated station and controller come from the RoboCurve YAM Isaac Lab Arena launchable. Dependencies include NumPy, Pillow, Shapely 2.x, Trimesh, Inspect Robots and the launchable's Isaac Lab packages.

Recorded environment: Isaac Lab `37ddf6268717`, Isaac Lab Arena `c7edf3a9d5d8`, and YAM Arena `0059d02a1a91`.
