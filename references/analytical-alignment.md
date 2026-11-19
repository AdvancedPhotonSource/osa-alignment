# Symmetry-guided OSA alignment

The controller uses brightness-reflection axes to infer the alignment point in
motor space. It starts at the actual readback, collects valid lines, and moves
to their least-squares intersection. Image evaluation is owned by
[`evaluate_alignment.py`](../scripts/evaluate_alignment.py), the symmetry fit by
[`symmetry.py`](../scripts/symmetry.py), and motor search by
[`align_osa.py`](../scripts/align_osa.py).

## Quantitative evaluation

The evaluator crops exclusive detector bounds `[x0, y0, x1, y1]`. Image coordinates
increase rightward and downward; fitted positions are relative to the crop.
RGB input is converted to luminance with weights `[0.2126, 0.7152, 0.0722]`.
Intensity values must be finite and nonnegative, with positive total signal.

With an empty crop, the first acquisition sets the detector bounds for the run.
The evaluator computes the center of mass of the full unsmoothed grayscale frame
and tests squares of side 128, 256, 384, 512, … pixels around it. It selects the
first enclosing at least 95% of the frame's integrated intensity, including
background. Pixel rounding places the window center within half a pixel per axis.
Windows extending beyond the detector use zero padding. Each observation records
the resolved bounds in `metrics.crop_xyxy`.

Gaussian smoothing (default sigma 1.5 pixels) precedes geometry fitting. The
intensity-weighted centroid supplies the initial point through which candidate
axes pass. Polar resampling covers the complete crop, extending values outside
it by zero. The relevant signal should be contained in the crop.

For polar brightness P(r,φ), calculate the angular FFT F at every radius and
inverse-transform `sum_r r * F(r)**2`. This is circular self-convolution, with
radial area weighting. Its angular coordinate is twice the reflection-axis
angle. Select up to three local maxima separated by at least 15° in axis angle.
This operates on brightness patterns of any shape.

Each candidate undergoes bounded Powell refinement in angle (±10°) and normal
displacement (±10 pixels). The objective is

\[
E=\frac{\sum_p[J(p)-J(R(p))]^2}{\sum_p[J(p)^2+J(R(p))^2]},
\]

where J is the smoothed image and R reflects about the trial line. The fixed
Cartesian evaluation region contains the crop and its reflected support for all
allowed trial lines. This keeps the compared signal constant as the line moves.
Bilinear resampling approximates reflection; optimization uses a 2-pixel grid
and candidate comparison uses a 1-pixel grid. Powell minimizes `1 + E` so its
relative convergence tolerance remains meaningful near zero mismatch. Select
the lowest-mismatch result; failure of that selected fit stops evaluation.

The perpendicular axis passes through the fitted axis point closest to the
centroid. On the final unit-spaced, zero-padded grid, compute the absolute
reflection discrepancy for each orientation using the smoothed crop \(J\).

\[
D(R)=\frac{\sum_p |J(p)-J(R(p))|}{2\sum_p J(p)},\qquad
\Delta=D_\perp-D_{\rm axis}.
\]

Accept exactly when this signed gap exceeds `mismatch_relative_tolerance`
(default 0.10). Multiplying the image by a positive constant preserves both
discrepancies. Each observation reports `d_axis`, `d_perp`, and `intensity_gap`.

## Axis direction

Sample the raw grayscale crop along the whole in-crop segment of the fitted
line, at spacing no larger than one pixel, using bilinear interpolation. Fit
`intensity = slope * signed_distance + intercept` with ordinary least squares.
Choose the sign of the axis vector so fitted intensity increases in that
direction. The slope is measured in intensity units per pixel. A numerically
zero slope leaves the direction undefined and stops the run.

Image direction `(ux, uy)` maps to unit motor direction `(ux, -uy)`. In motor
coordinates, the 3 o'clock direction is `(vy, -vx)` for directed axis `(vx, vy)`.
Each accepted line passes through its acquisition readback `p_i`:

\[
 n_i^T p = b_i,\quad n_i=(-v_{iy},v_{ix}),\quad b_i=n_i^T p_i.
\]

The image-axis intercept only locates the overlay in detector pixels. It is not
converted into a motor displacement.

## Search and intersection

Acquire one image at the starting readback. After an ambiguous-axis rejection,
move `-reverse_step_um * v`. After acceptance, store the line and, if more lines
are required, move `lateral_step_um * (vy, -vx)`. Reacquire after each search move.
As soon as the required line count is reached, solve

\[
 p_* = \arg\min_p\sum_i(n_i^T p-b_i)^2
\]

and use `p_*` as the final move target. Unit normals make the residuals perpendicular
motor-space distances in micrometers. All accepted lines receive equal weight.
The solver centers motor coordinates for numerical conditioning and reports
rank, condition number, residuals, and RMS distance. Rank below two stops the
run because the intersection is not unique. Each search and intersection move
clips x and y independently to their configured motor bounds. The move plan
records the proposed and bounded targets. Intersection diagnostics use the
unconstrained fit, and readback verification uses the bounded target.

The controller reads position before and after every acquisition and move.
Actual readbacks anchor the stored lines. The final search acquisition and the
final motor readback are separate result fields. Completion consists of the
intersection move and readback verification.

## Alignment settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `units` | required: `"um"` | Motor coordinates and distances in micrometers |
| `bounds` | required | Absolute `[x_min,x_max]`, `[y_min,y_max]` |
| `valid_axes_required` | 6 | Number of accepted lines before solving; at least 2 |
| `reverse_step_um` | 5 | Reverse-axis displacement after rejection |
| `lateral_step_um` | 5 | Clockwise perpendicular displacement after acceptance |
| `max_acquisitions` | 20 | Total acquisition limit, including the starting image; at least the required line count |
| `readback_tolerance` | 0.001 | Euclidean target/readback and acquisition-drift tolerance in µm; also permits this per-axis margin when reading global bounds |
| `evaluator.crop_xyxy` | `[276,73,616,413]` | Integer exclusive detector bounds, or `[]` / `null` for automatic cropping |
| `evaluator.mismatch_relative_tolerance` | 0.10 | Reject Δ at or below this fraction |
| `evaluator.sigma` | 1.5 | Gaussian sigma in pixels for reflection fitting; 0 disables smoothing |
| `evaluator.angle_samples` | 720 | Samples over the polar angle range 0–360°; corresponding axis angles span 0–180° |
| `evaluator.angle_bound_deg` | 10 | Local angle half-width |
| `evaluator.offset_bound_px` | 10 | Local normal-displacement half-width |
| `evaluator.max_candidates` | 3 | Maximum separated angular peaks to refine |
| `evaluator.evaluation_step` | 2 | Cartesian grid stride during optimization; final scores use stride 1 |

The bundled configuration retains the instrument's existing readback tolerance
of 10 µm; set this to the tolerance appropriate for the actual motor readbacks.
The bundled bounds are `[-30,30]` on both axes. Settings are recorded in the log.

## Adapter interface and APS 2-ID-D QueueServer adapter

A backend provides `units = "um"`, `read_position()` returning `[x,y]`,
`move_absolute([x,y])` that waits for completion, and `acquire_image()` returning
a locally accessible image path. `--adapter MODULE:FACTORY` loads a factory
accepting the JSON from `--adapter-config`.

[`APS2IDDAdapter`](../scripts/aps2idd_adapter.py) uses the existing QueueServer
helpers and converts between its micrometer interface and millimeter motor
coordinates. Load these helpers in the worker startup namespace after beamline
startup initializes the devices:

```python
from aps2idd_qserver import read_osa_position, move_osa_absolute
from s2idd_uprobe.qserver.xeye_osa import acquire_xeye_image
```

Allow the move plan and read/acquisition functions in QueueServer permissions.
Use the configured beamline client environment, an idle worker, and image paths
shared between the worker and controller hosts. The adapter configuration is
[`aps2idd.adapter.json`](../scripts/aps2idd.adapter.json). `post_call_sleep` sets a
dwell after successful adapter calls; the camera helper receives
`acquisition_timeout`. The camera exposure/gain should remain fixed during a run.

```sh
uv run --project scripts --locked --group beamline python scripts/align_osa.py \
  --config scripts/alignment.json \
  --adapter aps2idd_adapter:create_adapter --adapter-config scripts/aps2idd.adapter.json \
  --log trajectory.jsonl --output result.json --webui
```

## Offline replay

A manifest contains `units: "um"` and `samples`, each with a two-element
`position` and an `image` path relative to the manifest. Run:

```sh
uv run --project scripts --locked python scripts/align_osa.py \
  --config scripts/alignment.json --manifest recorded.json --start X Y \
  --log replay.jsonl --output replay-result.json --webui
```

Replay requires an image at every exact acquisition position requested by the
workflow; repeated images at one position cycle through that position's files.
The final intersection move needs no recorded image because it performs no
acquisition. Missing search positions stop replay. Sparse regular scans will
usually lack the continuous positions required by the directed workflow.

## Live dashboard

`--webui` serves a read-only dashboard on loopback (default port 8765).
`--webui-port 0` chooses an available port. Access it locally or through an SSH
forward. It polls the JSONL log once per second. All motor control remains in
the controller.

The top row shows the latest cropped acquisition with its axis and brightness
arrow, plus accepted motor-space lines, the readback trajectory, and intersection.
The trajectory has equal motor-coordinate scales. Drag to pan, Shift-drag to
zoom, and Reset view to return to the full range. Acquisition history includes
accept/reject/failure, motor position, both mismatches, Δ, axis angle, fitted
intensity slope, direction, and reasons. Click a row to browse its image; Go to
live resumes following the latest acquisition. At exit, a self-contained HTML
snapshot embeds all available image previews beside the log.

## Logs and outcomes

JSONL records settings, acquisition attempts, observations, move plans, completed
moves, the intersection, faults, and completion. A fresh log path is required.
Image/fitting failures remain available in history when an image was acquired.

- `aligned`: enough valid lines were collected and the bounded intersection move passed
  readback verification.
- `acquisition_limit`: the image limit was reached with too few accepted axes;
  no further search move is issued.
- `blocked`: fitting/direction, readback, acquisition, bounds, or intersection
  failure; no recovery move is issued.

The result includes the actual `final_readback`, `final_observation`, accepted
`lines`, `intersection` diagnostics, and counts. A non-aligned run exits with code
2. Nearly parallel accepted lines can yield a large condition number; the value
is reported alongside the inferred intersection. The result describes this
geometric procedure's outcome, rather than a separate final-image assessment.
