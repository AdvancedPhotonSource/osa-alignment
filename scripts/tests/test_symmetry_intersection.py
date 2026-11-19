"""Directed search and least-squares intersection tests without instrument access."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from align_osa import Controller, Replay, Settings, intersect_lines, motor_direction, symmetry_line
from evaluate_alignment import Config, evaluate


class Backend:
    units = 'um'

    def __init__(self, start=(0., 0.)):
        self.p = np.array(start, float)
        self.moves = []
        self.acquisitions = 0

    def read_position(self):
        return self.p.copy()

    def move_absolute(self, target):
        self.moves.append(list(target))
        self.p = np.array(target)

    def acquire_image(self):
        self.acquisitions += 1
        return 'synthetic.tiff'


def settings(**kwargs):
    return Settings(**dict(dict(units='um', bounds=[[-100,100],[-100,100]], valid_axes_required=2), **kwargs))


def observation(direction=(1., 0.), valid=True, error=None):
    return {'image': 'synthetic.tiff', 'metrics': {
        'valid': valid, 'direction_xy': list(direction), 'error': error,
        'reasons': [] if valid else ['ambiguous symmetry'],
        'reflection_mismatch': .01, 'perpendicular_mismatch': .2 if valid else .0105,
        'relative_mismatch_difference': .95 if valid else .05,
        'profile_slope_intensity_per_px': 1., 'fit': None, 'crop_xyxy': [0,0,32,32]}}


def run(tmp_path, observations, **kwargs):
    backend = Backend()
    it = iter(observations)
    controller = Controller(settings(**kwargs), backend, tmp_path/'run.jsonl', lambda _: next(it))
    result = controller.run()
    return controller, backend, result


def test_defaults_and_configuration():
    s=Settings('um',[[-30,30],[-30,30]])
    s.validate()
    assert (s.valid_axes_required,s.max_acquisitions,s.reverse_step_um,s.lateral_step_um)==(6,20,5,5)
    assert s.beta == .5
    for changes in ({'valid_axes_required':1},{'max_acquisitions':1},{'reverse_step_um':0},
                    {'lateral_step_um':float('nan')},{'units':'mm'},{'bounds':[[1,0],[0,1]]},
                    {'beta':0},{'beta':1.01},{'beta':float('nan')}):
        with pytest.raises(ValueError): settings(**changes).validate()
    with pytest.raises(TypeError): settings(stage_1_optimizer='bo')


def test_reject_reverse_accept_clockwise_and_final_intersection(tmp_path):
    c,b,r=run(tmp_path,[observation(valid=False),observation((0,-1)),observation((1,-1))])
    assert r['status']=='aligned'
    np.testing.assert_allclose(b.moves,[[-5,0],[0,0],[-5,-5]],atol=1e-12)
    assert b.acquisitions==3 and len(c.lines)==2
    np.testing.assert_allclose(r['intersection']['position'],[-5,-5])
    assert r['intersection']['rms_distance_um'] < 1e-12
    assert c.history[0]['line'] is None
    assert [o['valid_axes_count'] for o in c.history]==[0,1,2]


def test_configured_step_distances(tmp_path):
    _,b,r=run(tmp_path,[observation(valid=False),observation((0,-1)),observation((1,-1))],
              reverse_step_um=7,lateral_step_um=3)
    np.testing.assert_allclose(b.moves,[[-7,0],[-4,0],[-7,-3]],atol=1e-12)
    assert r['status']=='aligned'


def test_momentum_uses_consecutive_accepted_readbacks(tmp_path):
    flat = observation()
    flat['metrics'].update(direction_xy=None, fit={'axis_direction_xy':[1,0]})
    c,b,r = run(tmp_path,[observation((0,-1)), observation((0,1),valid=False),
                          observation((0,1)), flat, observation()],
                bounds=[[-3,3],[-20,20]], valid_axes_required=4, beta=.25)
    assert r['status']=='aligned'
    first,rejected,second,third,last = c.history
    assert 'momentum_um' not in rejected
    np.testing.assert_allclose([o['position'] for o in c.history],[[0,0],[3,0],[3,5],[3,5],[3,10]])
    np.testing.assert_allclose([o['momentum_um'] for o in (first,second,third,last)],
                               [[0,0],[.75,1.25],[.5625,.9375],[.421875,1.953125]])
    np.testing.assert_allclose(second['lateral_direction'],[1,0])
    np.testing.assert_allclose(third['lateral_direction'],[0,1])


@pytest.mark.parametrize('axis', [(1,0),(-1,0)])
def test_momentum_tie_is_independent_of_brightness_sign(tmp_path,axis):
    c,_,r = run(tmp_path,[observation((0,-1)),observation(axis),observation()],
                valid_axes_required=3,beta=1.)
    assert r['status']=='aligned'
    np.testing.assert_allclose(c.history[1]['momentum_um'],[5,0])
    np.testing.assert_allclose(c.history[1]['lateral_direction'],[0,1])


def test_line_anchor_uses_actual_readback(tmp_path):
    class OffsetBackend(Backend):
        def move_absolute(self,target):
            super().move_absolute(target)
            self.p += [.1,.2]
    b=OffsetBackend()
    it=iter([observation((0,-1)),observation((1,-1))])
    c=Controller(settings(readback_tolerance=.3),b,tmp_path/'run',lambda _:next(it))
    r=c.run()
    np.testing.assert_allclose(c.lines[1]['point'],[5.1,.2])
    np.testing.assert_allclose(r['intersection']['position'],[0,-4.9])
    np.testing.assert_allclose(r['final_readback'],[.1,-4.7])


def test_budget_stops_without_last_reverse_move(tmp_path):
    c,b,r=run(tmp_path,[observation(valid=False) for _ in range(4)],max_acquisitions=4)
    assert r['status']=='acquisition_limit' and len(b.moves)==3 and b.acquisitions==4
    assert not c.lines and r['intersection'] is None
    np.testing.assert_allclose(r['final_readback'],[-15,0])


def test_reverse_direction_persists_until_next_valid_image(tmp_path):
    flat = {'image':'flat.tiff', 'metrics':evaluate(np.ones((32,32)), Config(crop_xyxy=(0,0,32,32)))}
    seq = [observation(valid=False), observation((-1,0),valid=False), flat,
           observation((0,-1)), observation((0,1),valid=False),
           observation((1,0),valid=False), observation()]
    c,b,r = run(tmp_path,seq)
    assert r['status']=='aligned'
    np.testing.assert_allclose(b.moves[:6],[[-5,0],[-10,0],[-15,0],[-10,0],[-10,5],[-10,10]])
    assert [o.get('reverse_direction') for o in c.history] == [
        [-1,0],[-1,0],[-1,0],None,[0,1],[0,1],None]


def test_flat_first_rejection_cannot_seed_reverse_direction(tmp_path):
    flat = observation(valid=False)
    flat['metrics']['direction_xy'] = None
    c,b,r = run(tmp_path,[flat])
    assert r['status']=='blocked' and not b.moves
    assert 'brightness direction' in c.history[0]['metrics']['error']


def test_nth_axis_at_budget_still_moves_to_intersection(tmp_path):
    _,b,r=run(tmp_path,[observation((0,-1)),observation((1,-1))],max_acquisitions=2)
    assert r['status']=='aligned' and len(b.moves)==2 and b.acquisitions==2


def test_fit_failure_stops_without_motion(tmp_path):
    _,b,r=run(tmp_path,[observation(valid=False,error='undefined direction')])
    assert r['status']=='blocked' and not b.moves and b.acquisitions==1
    assert 'undefined direction' in r['reason']


def test_parallel_lines_stop_without_final_motion(tmp_path):
    _,b,r=run(tmp_path,[observation(),observation()])
    assert r['status']=='blocked' and 'rank-deficient' in r['reason']
    assert len(b.moves)==1 and b.acquisitions==2


def test_search_clips_each_axis_and_reacquires_at_boundary(tmp_path):
    c,b,r=run(tmp_path,[observation(valid=False) for _ in range(3)],
              bounds=[[-2,2],[-2,2]],max_acquisitions=3)
    assert r['status']=='acquisition_limit' and b.moves==[[-2,0]]
    assert [o['position'] for o in c.history]==[[0,0],[-2,0],[-2,0]]
    plan=next(e for e in map(json.loads,c.log_path.read_text().splitlines()) if e['event']=='plan')
    assert plan['proposed_target']==[-5,0] and plan['target']==[-2,0]


def test_search_and_intersection_use_bounded_targets(tmp_path):
    _,b,r=run(tmp_path,[observation(),observation((1,-.05))],bounds=[[-10,10],[-4,10]])
    assert r['status']=='aligned'
    np.testing.assert_allclose(b.moves,[[0,-4],[10,0]],atol=1e-12)
    np.testing.assert_allclose(r['intersection']['position'],[80,0],atol=1e-12)
    np.testing.assert_allclose(r['final_readback'],[10,0],atol=1e-12)


@pytest.mark.parametrize('target', [[float('nan'),0],[0,float('inf')],[0]])
def test_invalid_move_targets_still_fail(tmp_path,target):
    backend=Backend()
    controller=Controller(settings(),backend,tmp_path/'run')
    with pytest.raises(RuntimeError,match='finite motor coordinates'):
        controller.move(target,'test')
    assert not backend.moves


def test_bad_readback_no_recovery_motion(tmp_path):
    class Stuck(Backend):
        def move_absolute(self,target): self.moves.append(target)
    b=Stuck()
    c=Controller(settings(),b,tmp_path/'run',lambda _:observation())
    r=c.run()
    assert r['status']=='blocked' and 'readback' in r['reason'] and len(b.moves)==1
    assert b.acquisitions==1


def test_intersection_minimizes_perpendicular_distances():
    lines=[symmetry_line([1,0],[0,1]),symmetry_line([3,0],[0,1]),symmetry_line([0,4],[1,0])]
    result=intersect_lines(lines)
    np.testing.assert_allclose(result['position'],[2,4])
    assert result['rms_distance_um']==pytest.approx(np.sqrt(2/3))


def test_mapping_all_quadrants():
    np.testing.assert_allclose(motor_direction([3,4]),[.6,-.8])
    np.testing.assert_allclose(motor_direction([-3,-4]),[-.6,.8])


def test_replay_does_not_snap_and_final_move_needs_no_image():
    replay=Replay({'units':'um','samples':[{'position':[0,0],'image':'recorded.tiff'}]},[0,0])
    assert replay.acquire_image()=='recorded.tiff'
    replay.move_absolute([.1,.2])
    with pytest.raises(ValueError,match='exact'):replay.acquire_image()
    np.testing.assert_allclose(replay.read_position(),[.1,.2])


def test_log_records_moves_and_line_equations(tmp_path):
    c,b,r=run(tmp_path,[observation((0,-1)),observation((1,-1))])
    events=[json.loads(row) for row in c.log_path.read_text().splitlines()]
    assert [e['event'] for e in events].count('move')==2
    assert events[-1]['result']['status']=='aligned'
    assert len([e for e in events if e['event']=='intersection'])==1
    with pytest.raises(FileExistsError):Controller(settings(),b,c.log_path)
