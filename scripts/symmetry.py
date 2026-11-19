"""Brightness reflection symmetry in detector pixels.

x increases rightward and y downward. Axis angles are clockwise from +x,
modulo 180 degrees. Polar angular self-convolution uses complex FFT(P)**2
and radial area weights. Its argument is twice the reflection-axis angle.
Gaussian smoothing and bilinear resampling are implementation approximations.
Normalized reflection mismatch is measured on a fixed zero-padded domain;
zero extension assumes the crop contains the relevant signal. Local Powell
refinement uses a 2-pixel grid and final selection uses a 1-pixel grid.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize


def fit_symmetry(image, sigma=1.5, angle_samples=720, angle_bound_deg=10,
                 offset_bound_px=10, max_candidates=3, evaluation_step=2):
    """Return the best axis, candidate fits, and the normalized FFT score curve.

    Zero extension assumes the crop contains the relevant brightness pattern.
    The fixed evaluation disk's bounding square includes the crop and every
    reflection for |d| <= offset_bound_px. A low mismatch indicates mirror agreement;
    it does not establish that the chosen axis is unique. Candidate refinement is
    local and is not a guarantee of the global angle-and-position optimum.
    """
    image = np.asarray(image, dtype=float)
    if (image.ndim != 2 or not np.isfinite(image).all()
            or (image < 0).any() or image.sum() <= 0):
        raise ValueError('Expected a finite, nonnegative 2-D image with positive intensity')
    smooth = gaussian_filter(image, sigma, mode='constant')
    yy, xx = np.indices(image.shape)
    center = np.array([(smooth * xx).sum(), (smooth * yy).sum()]) / smooth.sum()
    height, width = image.shape
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1],
                        [width - 1, height - 1]])
    radius = np.linalg.norm(corners - center, axis=1).max() + 2
    radii = np.arange(0.5, np.ceil(radius), 1.0)
    phi = np.arange(angle_samples) * (2 * np.pi / angle_samples)
    polar = map_coordinates(smooth,
                            [center[1] + radii[:, None] * np.sin(phi),
                             center[0] + radii[:, None] * np.cos(phi)],
                            order=1, mode='grid-constant', cval=0, prefilter=False)
    spectrum = np.fft.rfft(polar, axis=1)
    score = np.fft.irfft((radii[:, None] * spectrum**2).sum(axis=0), n=angle_samples)
    score /= (radii[:, None] * polar**2).sum()
    angles = np.arange(angle_samples) * np.pi / angle_samples
    peaks = np.flatnonzero((score >= np.roll(score, 1)) & (score >= np.roll(score, -1)))
    selected = []
    for index in peaks[np.argsort(score[peaks])[::-1]]:
        if all(abs((angles[index] - angles[j] + np.pi / 2) % np.pi - np.pi / 2)
               >= np.deg2rad(15) for j in selected):
            selected.append(int(index))
        if len(selected) == max_candidates:
            break

    evaluation_radius = radius + 2 * offset_bound_px

    def evaluation_grid(step):
        gx = np.arange(np.floor(center[0] - evaluation_radius),
                       np.ceil(center[0] + evaluation_radius) + 1, step)
        gy = np.arange(np.floor(center[1] - evaluation_radius),
                       np.ceil(center[1] + evaluation_radius) + 1, step)
        grid_x, grid_y = np.meshgrid(gx, gy)
        original = map_coordinates(smooth, [grid_y, grid_x], order=1,
                                   mode='grid-constant', cval=0, prefilter=False)
        return grid_x, grid_y, original

    coarse_grid = evaluation_grid(evaluation_step)
    full_grid = evaluation_grid(1)

    def reflect(angle, offset, grid):
        grid_x, grid_y, _ = grid
        nx, ny = -np.sin(angle), np.cos(angle)
        distance = nx * (grid_x - center[0]) + ny * (grid_y - center[1]) - offset
        return map_coordinates(smooth,
                                    [grid_y - 2 * ny * distance,
                                     grid_x - 2 * nx * distance],
                                    order=1, mode='grid-constant', cval=0, prefilter=False)
    def mismatch(angle, offset, grid):
        original = grid[2]
        reflected = reflect(angle, offset, grid)
        return float(np.sum((original - reflected)**2)
                     / (np.sum(original**2) + np.sum(reflected**2)))

    candidates = []
    for index in selected:
        initial_angle = angles[index]
        # Scale both optimization coordinates to [-1, 1].
        angle_scale = np.deg2rad(angle_bound_deg)
        # The constant makes Powell's relative ftol an absolute mismatch tolerance
        # near perfect symmetry, where the mismatch approaches zero.
        fit = minimize(lambda p: 1.0 + mismatch(initial_angle + p[0] * angle_scale,
                                                p[1] * offset_bound_px, coarse_grid),
                       [0.0, 0.0], method='Powell', bounds=[(-1, 1), (-1, 1)],
                       options={'xtol': 1e-3, 'ftol': 1e-8, 'maxfev': 1000})
        angle = initial_angle + fit.x[0] * angle_scale
        offset = fit.x[1] * offset_bound_px
        initial_error = mismatch(initial_angle, 0, full_grid)
        refined_error = mismatch(angle, offset, full_grid)
        normal = np.array([-np.sin(angle), np.cos(angle)])
        point = center + offset * normal
        canonical_angle = angle % np.pi
        canonical_normal = np.array([-np.sin(canonical_angle), np.cos(canonical_angle)])
        candidates.append({
            'initial_angle_deg': float(np.rad2deg(initial_angle)),
            'initial_fft_score': float(score[index]),
            'initial_mismatch': initial_error,
            'angle_deg': float(np.rad2deg(canonical_angle)),
            'offset_px': float((point - center) @ canonical_normal),
            'axis_point_xy_px': point.tolist(),
            'axis_direction_xy': [float(np.cos(canonical_angle)), float(np.sin(canonical_angle))],
            'mismatch': refined_error,
            'optimizer_success': bool(fit.success),
            'optimizer_message': str(fit.message),
            'optimizer_evaluations': int(fit.nfev),
            'at_search_bound': bool(np.any(np.abs(fit.x) > 0.99)),
        })
    candidates.sort(key=lambda item: item['mismatch'])
    best = candidates[0]
    if not best['optimizer_success']:
        raise RuntimeError(f"Symmetry refinement failed: {best['optimizer_message']}")
    angle = np.deg2rad(best['angle_deg'])
    point = np.array(best['axis_point_xy_px'])
    perpendicular_angle = angle + np.pi / 2
    perpendicular_normal = np.array([-np.sin(perpendicular_angle), np.cos(perpendicular_angle)])
    perpendicular_offset = float((point - center) @ perpendicular_normal)
    original = full_grid[2]
    along = reflect(angle, best['offset_px'], full_grid)
    across = reflect(perpendicular_angle, perpendicular_offset, full_grid)
    best['perpendicular_mismatch'] = float(np.sum((original - across)**2)
                                          / (np.sum(original**2) + np.sum(across**2)))
    # Integrate absolute residuals over the full zero-padded pixel grid.
    normalization = 2 * smooth.sum()
    best['d_axis'] = float(np.abs(original - along).sum() / normalization)
    best['d_perp'] = float(np.abs(original - across).sum() / normalization)
    return {
        'centroid_xy_px': center.tolist(),
        'best_axis': candidates[0],
        'candidates': candidates,
        'fft_score_range': float(np.ptp(score)),
        'fft_angles_deg': np.rad2deg(angles).tolist(),
        'fft_scores': score.tolist(),
    }

