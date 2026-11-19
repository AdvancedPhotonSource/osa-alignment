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
   against signed pixel distance. Orient the axis toward increasing fitted
   intensity.
3. Compare reflection mismatch E about this axis with E⊥ about its perpendicular
   through the closest axis point to the intensity centroid. Reject if
   `abs(E⊥ − E) / max(E⊥, E) <= mismatch_relative_tolerance` (default 0.10).
   Both-zero mismatch also rejects. After rejection, move opposite the directed
   axis by `reverse_step_um` (default 5).
4. For an accepted axis, store a motor-space line through the acquisition's
   actual readback, oriented by the mapped image axis. Move to the axis's
   3 o'clock side by `lateral_step_um` (default 5).
5. As soon as `valid_axes_required` lines have been stored (default 6), solve
   their unweighted least-squares intersection and move directly there. This
   replaces the lateral move after the last required axis and completes the run.

Stop at the current position on an acquisition/fitting/direction failure,
rank-deficient intersection, motor fault, or out-of-bounds target. Stop after
`max_acquisitions` images (default 20) if too few axes were accepted. These stops
perform no additional motor move. A successful result reports the intersection
and final readback; the most recent image is the last search acquisition.

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
