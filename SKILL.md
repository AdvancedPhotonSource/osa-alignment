---
name: osa-alignment
description: Align an order sorting aperture using directed brightness-reflection axes and the least-squares intersection of their motor-space lines. Use for automated OSA alignment, including strongly distorted images without a recognizable ring.
---

# OSA alignment

Run the symmetry-guided controller in [scripts/align_osa.py](scripts/align_osa.py).
The procedure uses the brightness pattern in a configured detector crop. Its
image-to-motor mapping is image right → motor +x and image down → motor −y.
Motor positions and step distances are in micrometers.

## Workflow

1. Acquire at the actual starting readback and crop the image. Locate the
   brightness-reflection axis using polar angular FFT self-convolution, then
   refine its angle and perpendicular displacement locally.
2. Sample the original pixel intensities along the fitted axis and fit a line
   against signed pixel distance to measure the brightness direction.
3. Compute the absolute reflection residual for the fitted axis and its
   perpendicular through the closest axis point to the intensity centroid.
   Normalize each residual sum by twice the smoothed crop's integrated intensity.
   Reject if `D_perp − D_axis <= mismatch_relative_tolerance` (default 0.05).
   On the first rejection, move opposite the measured brightness direction by
   `reverse_step_um` (default 5). Reuse that direction for consecutive
   rejections. The next valid image ends the streak.
4. For an accepted axis, store a motor-space line through the acquisition's
   actual readback. The first accepted axis seeds the lateral move on the
   clockwise side of the brightness direction. From the second accepted axis
   onward, update momentum using successive accepted readback displacements
   (exponential weight `beta`, default 0.5). Choose the perpendicular that
   points along that momentum and move `lateral_step_um` (bundled value 15 µm).
5. As soon as `valid_axes_required` lines have been stored (default 6), solve
   their unweighted least-squares intersection and use it as the move target.
   In adapter runs, acquire one final image after verifying the motor readback.

Clip each search and intersection move per axis to the configured motor bounds.
Stop at the current position on an acquisition/fitting/direction failure,
rank-deficient intersection, or motor fault. Stop after
`max_acquisitions` search images (default 20) if too few axes were accepted.
These stops perform no additional motor move. A successful adapter run reports
the intersection, final readback, and final image. The final image is for visual
inspection and is separate from the search budget.

## Run

Set absolute motor `bounds`, `readback_tolerance`, and `evaluator.crop_xyxy` in
[scripts/alignment.json](scripts/alignment.json). The supplied crop is
`[276, 73, 616, 413]`, with exclusive upper bounds, matching the recorded study.
Set `evaluator.crop_xyxy` to `[]` or `null` to center a square on the first image's
intensity centroid. Its side starts at 128 pixels and grows in 128-pixel steps
until it contains at least 95% of that frame's total intensity. The run reuses
these bounds for every image.
The supplied adapter uses the APS 2-ID-D QueueServer; acquire and readback calls
must return locally accessible image paths and coordinates in micrometers.

```sh
uv sync --project scripts --locked --group beamline
uv run --project scripts --locked --group beamline python scripts/align_osa.py \
  --config scripts/alignment.json \
  --adapter aps2idd_adapter:create_adapter --adapter-config scripts/aps2idd.adapter.json \
  --log trajectory.jsonl --output result.json --webui
```

Use a fresh log path for every run. `--webui-port 0` selects an available local
port. Share the printed URL while the controller runs. The dashboard shows the
latest crop and directed symmetry axis at the top, accepted motor-space lines,
the move trajectory, intersection, and acquisition history. Drag the search
plot to pan; Shift-drag selects a zoom region; Reset view restores the full view.
Click an acquisition to inspect its image and Go to live to follow the run.
A self-contained HTML snapshot is saved beside the JSONL log at exit.

To evaluate a single image:

```sh
uv run --project scripts --locked python scripts/evaluate_alignment.py IMAGE \
  --crop 276 73 616 413 --output symmetry.json
```

See [the analytical guide](references/analytical-alignment.md) for the numerical
contracts, settings, adapter interface, exact-position replay, and result fields.
