---
name: osa-alignment
description: Automate order sorting aperture (OSA) alignment from reflection symmetry in detector images, including strongly distorted patterns without a recognizable ring.
---

# OSA alignment

The controller in [scripts/align_osa.py](scripts/align_osa.py) estimates the
motor position that aligns the order sorting aperture (OSA) from brightness
patterns acquired at several positions. It analyzes a rectangular region of
each detector image, referred to as the crop. The motor readback is the
position reported by the motors. Positions and step distances are in
micrometers (µm). When mapping image directions to motor directions, image
right corresponds to positive motor x and image down to negative motor y.

## Workflow

1. Acquire the first image at the starting readback. Smooth the crop and
   compute its intensity centroid, the mean pixel position weighted by
   intensity. A brightness-reflection axis is a line about which the intensity
   pattern is approximately mirror symmetric. To find candidate axes through
   the centroid, resample the smoothed crop in radius and angle and convolve
   each angular profile with itself using a fast Fourier transform (FFT).
   Refine each candidate by varying its angle and perpendicular offset to
   minimize the sum of squared differences between the smoothed crop and its
   reflection, normalized by the sum of squared intensities in the two images.
   Keep the candidate with the lowest discrepancy.
2. Sample the unsmoothed crop along the fitted axis. Fit a straight line to
   intensity as a function of signed distance along the axis in pixels. The
   direction in which the fitted intensity increases is the brightness
   direction. A numerically flat profile leaves this direction undefined.
3. To determine whether the fitted axis is more strongly supported than its
   perpendicular, compare the reflections about both lines. Place the
   perpendicular through the point on the fitted axis nearest the intensity
   centroid. For each line, sum the absolute differences between the smoothed
   crop and its reflection, treating the intensity outside the crop as zero.
   Divide by twice the total intensity of the smoothed crop. Let `D_axis` and
   `D_perp` denote the resulting dimensionless discrepancies for the fitted
   axis and the perpendicular, respectively. The setting
   `evaluator.mismatch_relative_tolerance` gives the acceptance threshold
   (default 0.05). Accept the axis when `D_perp - D_axis` exceeds this threshold.
4. For an accepted axis, map the brightness direction to motor coordinates,
   using the fitted axis orientation if the brightness profile is flat. Store
   a line parallel to that direction through the readback measured immediately
   after the acquisition. The setting
   `valid_axes_required` gives the number of lines needed to estimate the
   alignment position (default 6). After collecting this many lines, use the
   point that minimizes the sum of squared perpendicular distances to all
   lines as the target. This point is their unweighted least-squares intersection.
5. The setting `max_acquisitions` limits the number of images used in the
   search, including the starting image (default 20). If more lines are needed
   and the limit has yet to be reached, move and repeat the analysis on the
   image acquired at the new position.

   - **Rejected axis.** On the first rejection, move opposite the measured
     brightness direction by the reverse step distance `reverse_step_um`
     (default 5 µm). Keep this direction through consecutive rejections. An
     accepted axis clears it so the next rejection determines a new direction.
   - **Accepted axis.** On the first acceptance, move along the clockwise
     perpendicular to the mapped axis direction. To steer later steps, track
     an exponentially weighted average of the displacements between accepted
     readbacks, referred to as momentum. Start at zero and give the newest
     displacement a weight of `beta` (default 0.5). From the second acceptance
     onward, choose the perpendicular direction with the larger projection onto
     this average. The lateral step distance is `lateral_step_um` (15 µm in
     the supplied configuration).

The configured motor bounds are the lower and upper limits on each axis. The
controller limits each component of a move target independently to these
bounds. The run ends if the search limit is reached with too few accepted
axes, the lines yield no unique intersection, or acquisition, fitting,
direction measurement, or motor control fails. A flat profile causes a
direction failure when the first rejection requires a new reverse direction.

A hardware adapter connects the controller to the motors and detector. In a
successful adapter run, the controller verifies the readback against the
bounded intersection target and acquires one final image for visual inspection
outside the search limit. The result reports the calculated intersection, the
final readback, and the final image path.

## Run

In [scripts/alignment.json](scripts/alignment.json), set `bounds` to the absolute
lower and upper limits for each motor axis. Set `readback_tolerance` to the
allowed distance in micrometers between the bounded target and the reported
position. This setting also limits position drift during acquisition. Set
`evaluator.crop_xyxy` to the left, top, right, and bottom pixel bounds, with the
right and bottom bounds excluded. The supplied bounds are `[276, 73, 616, 413]`,
matching the recorded study.

Set `evaluator.crop_xyxy` to `[]` or `null` to determine the crop from the first
image. The controller centers a square on the intensity centroid of the full
unsmoothed frame and starts with a side length of 128 pixels. It increases the
side length in 128-pixel increments until the square contains at least 95% of
the total intensity, then uses the same crop for every image in the run.

The supplied adapter uses the Advanced Photon Source (APS) 2-ID-D QueueServer.
It returns locally accessible image paths and motor readbacks in micrometers.
Pass the alignment settings to `--config` and select the adapter factory and
its settings with `--adapter` and `--adapter-config`. Use a fresh path for the
JSON Lines log (`--log`) on each run and set `--output` to the result file.
The `--webui` option starts a live dashboard.

```sh
uv sync --project scripts --locked --group beamline
uv run --project scripts --locked --group beamline python scripts/align_osa.py \
  --config scripts/alignment.json \
  --adapter aps2idd_adapter:create_adapter --adapter-config scripts/aps2idd.adapter.json \
  --log trajectory.jsonl --output result.json --webui
```

To select an available local dashboard port, use `--webui-port 0`. Share the
printed URL while the controller runs. The dashboard shows the latest crop
with the fitted axis and brightness direction, the accepted motor
lines, the trajectory, the intersection, and the acquisition history. Drag the
plot to pan, Shift-drag to select a zoom region, and use **Reset view** to
restore the full view. Select an acquisition to inspect its image or use
**Go to live** to follow the run. At exit, the dashboard saves a self-contained
HTML snapshot alongside the log.

To evaluate a single image, replace `IMAGE` with the path to the detector image.
The `--crop` argument takes the same four pixel bounds as `evaluator.crop_xyxy`.

```sh
uv run --project scripts --locked python scripts/evaluate_alignment.py IMAGE \
  --crop 276 73 616 413 --output symmetry.json
```

See [the analytical guide](references/analytical-alignment.md) for the numerical
methods, settings, adapter interface, and result fields. The guide also describes
offline replay, which uses recorded images at the exact requested motor positions.
