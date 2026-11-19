"""Numerical contracts for cropping, FFT symmetry, direction, and acceptance."""
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_alignment import Config, auto_crop_bounds, axis_segment, crop_image, directed_axis, evaluate, read_image
from symmetry import fit_symmetry
from test_symmetry_intersection import Backend, Controller, settings


def pattern(angle_deg, center=(47.2,48.6)):
    yy,xx=np.indices((96,96))
    angle=np.deg2rad(angle_deg)
    u=(xx-center[0])*np.cos(angle)+(yy-center[1])*np.sin(angle)
    v=-(xx-center[0])*np.sin(angle)+(yy-center[1])*np.cos(angle)
    return (200*np.exp(-((u-8)**2/70+v*v/40))
            +90*np.exp(-((u+10)**2/45+(v-10)**2/12))
            +90*np.exp(-((u+10)**2/45+(v+10)**2/12)))


def config(**kwargs):
    return Config(**dict(dict(crop_xyxy=(0,0,96,96)),**kwargs))


@pytest.mark.parametrize('core,wing,offset,side', [
    (190, 5, 160, 128), (188, 6, 100, 256), (188, 6, 160, 384), (188, 6, 200, 512),
])
def test_auto_crop_includes_threshold_and_grows_by_128(core, wing, offset, side):
    image = np.zeros((600, 720))
    image[280, 400] = core
    image[280, [400 - offset, 400 + offset]] = wing
    x0, y0, x1, y1 = auto_crop_bounds(image)
    assert (x1-x0, y1-y0) == (side, side)
    assert abs((x0+x1-1)/2 - 400) <= .5
    assert abs((y0+y1-1)/2 - 280) <= .5


def test_auto_crop_preserves_edge_signal_and_square():
    image = np.zeros((40, 70))
    image[3, 4] = 90
    image[3, 14] = 10
    bounds = auto_crop_bounds(image)
    patch = crop_image(image, bounds, pad=True)
    assert patch.shape == (128, 128)
    assert patch.sum() == image.sum()
    assert patch[3-bounds[1], 4-bounds[0]] == 90
    assert abs(bounds[0] + 63.5 - 5) <= .5
    assert abs(bounds[1] + 63.5 - 3) <= .5


@pytest.mark.parametrize('empty', [[], None])
def test_controller_reuses_first_auto_crop(tmp_path, empty):
    class Images(Backend):
        def acquire_image(self):
            frame = np.zeros((320, 420), dtype=np.float32)
            start = 100 + 30*self.acquisitions
            frame[100:196, start:start+96] = pattern(26)
            path = tmp_path / f'{self.acquisitions}.tif'
            Image.fromarray(frame).save(path)
            self.acquisitions += 1
            return path

    run = Controller(settings(evaluator={'crop_xyxy': empty}), Images(), tmp_path/'run.jsonl')
    first, second = run.acquire(), run.acquire()
    assert first['metrics']['error'] is None and second['metrics']['error'] is None
    assert second['metrics']['crop_xyxy'] == first['metrics']['crop_xyxy']
    assert auto_crop_bounds(read_image(second['image'])) != first['metrics']['crop_xyxy']


def test_auto_crop_zero_signal_is_measurement_failure():
    result = evaluate(np.zeros((32, 32)), Config(crop_xyxy=[]))
    assert not result['valid'] and 'positive sum' in result['error']


@pytest.mark.parametrize('angle',[0,37.3,90,179.4])
def test_nonring_axis_recovery_and_brightness_direction(angle):
    result=evaluate(pattern(angle),config())
    assert result['valid'],result['reasons']
    a=np.deg2rad(angle)
    direction=np.array([np.cos(a),np.sin(a)])
    assert np.array(result['direction_xy']) @ direction > .9999
    fit=result['fit']
    error=abs((fit['angle_deg']-angle+90)%180-90)
    assert error<.15 and fit['mismatch']<.001
    normal=np.array([-direction[1],direction[0]])
    assert abs((np.array(fit['axis_point_xy_px'])-[47.2,48.6]) @ normal)<.1
    assert fit['perpendicular_mismatch'] > .05


def test_direction_is_ols_slope_on_raw_profile():
    yy,xx=np.indices((32,40))
    image=10+2*xx.astype(float)
    for vector in ([1.,0.],[-1.,0.]):
        result=directed_axis(image,[12.,10.],vector)
        np.testing.assert_allclose(result['direction_xy'],[1,0])
        assert abs(result['profile_slope_intensity_per_px'])==pytest.approx(2)
    with pytest.raises(ValueError,match='undefined'):
        directed_axis(np.ones((32,40)),[12,10],[1,0])


def test_axis_segment_vertical_and_outside():
    assert axis_segment([12,10],[0,1],(32,40))==(-10,21)
    with pytest.raises(ValueError,match='intersect'):
        axis_segment([100,10],[0,1],(32,40))


@pytest.mark.parametrize('axis,perpendicular,tolerance,valid',[(.9,1,.1,False),(.8,1,.1,True),(0,0,.1,False),(1,.9,.11,False),(.8,1,.25,False)])
def test_relative_difference_rule(monkeypatch,axis,perpendicular,tolerance,valid):
    import evaluate_alignment as module
    fit={'axis_point_xy_px':[47,47],'axis_direction_xy':[1,0],
         'mismatch':axis,'perpendicular_mismatch':perpendicular,'angle_deg':0}
    monkeypatch.setattr(module,'fit_symmetry',lambda *a,**k:dict(best_axis=fit,centroid_xy_px=[47,47],fft_score_range=0,candidates=[fit]))
    _,xx=np.indices((96,96))
    result=evaluate(1+xx.astype(float),config(mismatch_relative_tolerance=tolerance))
    assert result['valid'] is valid
    assert not result['error']


def test_crop_preserves_geometry_in_crop_coordinates(tmp_path):
    image=np.pad(pattern(26),((10,5),(20,15)))
    path=tmp_path/'image.tiff'
    Image.fromarray(image.astype(np.float32)).save(path)
    result=evaluate(read_image(path),config(crop_xyxy=(20,10,116,106)))
    assert result['crop_shape']==[96,96] and result['image_shape']==[111,131]
    assert result['valid']
    assert abs(result['fit']['angle_deg']-26)<.15
    assert result['centroid_xy_px'][0]<96


def test_failures_have_no_fabricated_axis_or_direction():
    result=evaluate(np.zeros((96,96)),config())
    assert not result['valid'] and result['error'] and 'direction_xy' not in result
    result=evaluate(pattern(0),Config())
    assert not result['valid'] and 'crop' in result['error']
    with pytest.raises(ValueError):config(crop_xyxy=(0,0,1,1)).validate()


def test_fft_matches_direct_circular_self_convolution():
    rng=np.random.default_rng(10)
    polar=rng.random((4,32));radii=np.arange(.5,4)
    f=np.fft.rfft(polar,axis=1)
    fft=np.fft.irfft((radii[:,None]*f**2).sum(axis=0),n=32)
    direct=np.array([sum(radii[r]*sum(polar[r,j]*polar[r,(k-j)%32] for j in range(32)) for r in range(4)) for k in range(32)])
    np.testing.assert_allclose(fft,direct,atol=1e-12)


def test_refinement_corrects_centroid_bias_and_intensity_scale():
    image=pattern(37.3)
    yy,xx=np.indices(image.shape)
    image+=20*np.exp(-((xx-25)**2+(yy-75)**2)/8)
    a=fit_symmetry(image)
    b=fit_symmetry(image*8)
    best=a['best_axis']
    assert best['mismatch']<best['initial_mismatch']
    assert abs(best['offset_px'])>.05
    assert abs((best['angle_deg']-37.3+90)%180-90)<.5
    assert best['angle_deg']==pytest.approx(b['best_axis']['angle_deg'],abs=1e-5)


def test_actual_image_closed_loop(tmp_path):
    optimum=np.array([2.,-3.])
    class ImageBackend(Backend):
        def acquire_image(self):
            self.acquisitions+=1
            motor=optimum-self.p
            angle=np.rad2deg(np.arctan2(-motor[1],motor[0]))
            path=tmp_path/f'image-{self.acquisitions}.tiff'
            Image.fromarray(pattern(angle).astype(np.float32)).save(path)
            return str(path)
    backend=ImageBackend(start=(-12.,-9.))
    c=Controller(settings(valid_axes_required=6,evaluator={'crop_xyxy':[0,0,96,96]}),backend,tmp_path/'run.jsonl')
    result=c.run()
    assert result['status']=='aligned',result['reason']
    assert result['valid_axes_count']==6 and backend.acquisitions==6
    np.testing.assert_allclose(result['final_readback'],optimum,atol=.1)
