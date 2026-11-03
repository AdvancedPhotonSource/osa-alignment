"""M2 optionally penalizes hole intensity that sits off the inner ellipse center."""
import json
import sys

import numpy as np
import pytest
from PIL import Image

from test_analytical_alignment import Config, ring_image
import evaluate_alignment as module

AXES = dict(inner_axes=(12, 12), outer_axes=(30, 30))


def filled_ring(fill, offset=0):
    """Return ring_image() with its hole filled, optionally as an off-center blob."""
    image = ring_image()
    y, x = np.indices(image.shape)
    u, v = x - 50., y - 50.
    hole = (u / 12)**2 + (v / 12)**2 < 1
    image[hole] = 1.
    if offset:
        image[hole & ((u - offset)**2 + v**2 < 9)] = fill
    else:
        image[hole] = fill
    return image


def hole_mask(result, shape):
    """Rebuild the evaluator's hole from the reported geometry, as the metric does."""
    fit = result['fit']
    y, x = np.indices(shape)
    dx, dy = x - fit['center_x'], y - fit['center_y']
    angle = np.deg2rad(fit['angle_degrees'])
    u = np.cos(angle) * dx + np.sin(angle) * dy
    v = -np.sin(angle) * dx + np.cos(angle) * dy
    a, b = fit['inner_axes']
    return (u / a)**2 + (v / b)**2 < 1


@pytest.mark.parametrize('fixed', [False, True])
def test_zero_weight_uses_the_central_leakage_ratio(fixed):
    image = filled_ring(30, offset=6)
    axes = AXES if fixed else {}
    result = module.evaluate(image, Config(**axes, intensity_normalizer=5,
                                           hole_asymmetry_weight=0))
    assert result['valid'], result['reasons']
    expected = float(image[hole_mask(result, image.shape)].sum())
    assert result['hole_integral'] == expected
    assert result['secondary_metric'] == pytest.approx(
        result['hole_weighted_mean'] / result['annulus_mean'] / 5)
    assert result['hole_centroid_offset_px'] > 0


def test_centered_hole_intensity_is_not_penalized():
    """A uniform hole leaves only the sub-pixel centroid floor of a discretized disc."""
    result = module.evaluate(filled_ring(30), Config(**AXES, hole_asymmetry_weight=2))
    assert result['valid'], result['reasons']
    assert result['hole_centroid_offset_px'] < 0.25
    assert result['secondary_metric'] == pytest.approx(
        result['hole_weighted_mean'] / result['annulus_mean'], rel=0.05)


def test_optional_asymmetry_raises_m2_above_the_central_leakage_ratio():
    centered = module.evaluate(filled_ring(30), Config(**AXES))
    offset = module.evaluate(filled_ring(200, offset=6), Config(**AXES, hole_asymmetry_weight=2))
    assert offset['valid'], offset['reasons']
    assert offset['hole_centroid_offset_px'] > 1
    assert offset['secondary_metric'] > offset['hole_weighted_mean'] / offset['annulus_mean']
    assert offset['hole_centroid_offset_px'] > centered['hole_centroid_offset_px']


def test_offset_fraction_scales_the_penalty():
    image = filled_ring(200, offset=6)
    result = module.evaluate(image, Config(**AXES, hole_asymmetry_weight=2))
    fraction = result['hole_centroid_offset_fraction']
    assert fraction == pytest.approx(result['hole_centroid_offset_px'] / 12, rel=1e-9)
    assert result['secondary_metric'] == pytest.approx(
        result['hole_weighted_mean'] / result['annulus_mean'] * (1 + 2 * fraction), rel=1e-9)


def test_dark_hole_keeps_a_finite_defined_score():
    """A perfectly dark hole has no centroid; it is the ideal, not a failure.

    Inner semiaxes of 10 stay clear of the ring's r >= 12 rim, so the hole is
    genuinely empty and its weighted centroid is a 0/0 that must not become NaN.
    """
    result = module.evaluate(ring_image(), Config(inner_axes=(10, 10), outer_axes=(30, 30)))
    assert result['valid'], result['reasons']
    assert result['hole_integral'] == 0
    assert result['hole_centroid_offset_px'] == 0
    assert result['secondary_metric'] == 0
    json.dumps(module.json_safe(result), allow_nan=False)


def test_overcorrected_background_cannot_diverge_the_centroid():
    """Clipped weights keep the centroid finite where a raw (I - B) sum crosses zero."""
    image = filled_ring(200, offset=6)
    radius = np.hypot(*np.indices(image.shape)[::-1] - 50.)
    inside = image[radius < 12]
    offsets = [module.evaluate(image, Config(**AXES, background=b))['hole_centroid_offset_px']
               for b in (0, inside.mean(), inside.max() + 10)]
    assert all(np.isfinite(o) and o < 12 for o in offsets)


@pytest.mark.parametrize('weight', [-0.1, -1, np.nan, np.inf])
def test_invalid_hole_asymmetry_weight(weight):
    with pytest.raises(ValueError, match='hole_asymmetry_weight'):
        Config(hole_asymmetry_weight=weight).validate()


def test_failed_fits_keep_infinite_m2(monkeypatch):
    monkeypatch.setattr(module, 'canny', lambda image, **kwargs: np.zeros_like(image, dtype=bool))
    result = module.evaluate(filled_ring(200, offset=6), Config(**AXES))
    assert not result['valid']
    assert np.isinf(result['secondary_metric'])


def test_cli_hole_asymmetry_option(tmp_path, monkeypatch, capsys):
    image = tmp_path / 'ring.tiff'
    Image.fromarray(filled_ring(200, offset=6).astype(np.float32)).save(image)
    monkeypatch.setattr(sys, 'argv', ['evaluate_alignment.py', str(image),
                                      '--hole-asymmetry-weight', '3'])
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert result['config']['hole_asymmetry_weight'] == 3
    assert result['hole_centroid_offset_px'] > 0


def test_cli_defaults_to_the_documented_weight(tmp_path, monkeypatch, capsys):
    image = tmp_path / 'ring.tiff'
    Image.fromarray(ring_image().astype(np.float32)).save(image)
    monkeypatch.setattr(sys, 'argv', ['evaluate_alignment.py', str(image)])
    module.main()
    assert json.loads(capsys.readouterr().out)['config']['hole_asymmetry_weight'] == 0.
