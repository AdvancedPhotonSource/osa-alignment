"""Central weighting and ring normalization are always active in M2."""
import numpy as np
import pytest
from PIL import Image

from test_analytical_alignment import Controller, SyntheticBackend, settings
from test_hole_asymmetry import AXES, filled_ring
from evaluate_alignment import Config, evaluate


def test_default_disables_the_centroid_penalty():
    result = evaluate(filled_ring(200, offset=6), Config(**AXES))
    assert result['valid'], result['reasons']
    assert result['hole_centroid_offset_fraction'] > 0
    assert result['config']['hole_asymmetry_weight'] == 0
    assert result['secondary_metric'] == pytest.approx(
        result['hole_weighted_mean'] / result['annulus_mean'])


def test_controller_uses_central_leakage_with_symmetry_disabled(tmp_path):
    path = tmp_path / 'ring.tiff'
    Image.fromarray(filled_ring(200, offset=6).astype(np.float32)).save(path)
    controller = Controller(settings(evaluator=AXES), SyntheticBackend([0, 0]), tmp_path / 'log')
    result = controller.measurement(path)['metrics']
    assert result['valid'], result['reasons']
    assert result['config']['hole_asymmetry_weight'] == 0
    assert result['hole_centroid_offset_fraction'] > 0
    assert result['secondary_metric'] == pytest.approx(
        result['hole_weighted_mean'] / result['annulus_mean'])


@pytest.mark.parametrize('gain,pedestal', [(1., 25.), (3., 0.), (3., 25.)])
def test_ring_normalization_cancels_gain_and_corrected_background(gain, pedestal):
    image = filled_ring(30)
    reference = evaluate(image, Config(**AXES))
    scaled = evaluate(gain*image + pedestal, Config(**AXES, background=pedestal))
    assert reference['valid'] and scaled['valid']
    assert reference['secondary_metric'] == pytest.approx(.3, rel=.02)
    assert scaled['secondary_metric'] == pytest.approx(reference['secondary_metric'], rel=1e-6)
    assert scaled['primary_metric'] == pytest.approx(reference['primary_metric'], rel=1e-6)


def test_central_signal_outweighs_equal_signal_near_the_rim():
    image = filled_ring(10)
    central, rim = image.copy(), image.copy()
    central[49:52, 49:52] += 10
    rim[49:52, 57:60] += 10
    assert central.sum() == rim.sum()
    baseline, central_result, rim_result = [evaluate(im, Config(**AXES))
                                            for im in [image, central, rim]]
    assert all(r['valid'] for r in [baseline, central_result, rim_result])
    central_increase = central_result['secondary_metric'] - baseline['secondary_metric']
    rim_increase = rim_result['secondary_metric'] - baseline['secondary_metric']
    assert central_increase > 5 * rim_increase > 0
    assert np.isfinite(central_result['secondary_metric'])
