"""Read-only symmetry dashboard, overlays, acquisition history, and CLI lifecycle."""
import base64
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import align_osa
from alignment_webui import Dashboard
from test_symmetry_intersection import Backend, observation, settings


def request(dashboard,path='api/status',method='GET'):
    return urlopen(Request(dashboard.url+path,method=method),timeout=5)


def image_observation(path,size=32,valid=True):
    obs=observation((0,1),valid)
    obs['image']=str(path)
    obs['metrics'].update(crop_xyxy=[0,0,size,size],fit={
        'axis_point_xy_px':[size/2,size/2],'axis_direction_xy':[0,1],'angle_deg':90})
    return obs


def test_live_state_read_only_routes_and_moves(tmp_path):
    path=tmp_path/'frame.tiff'
    Image.fromarray(np.arange(1024,dtype=np.uint16).reshape(32,32)).save(path)
    backend=Backend()
    c=align_osa.Controller(settings(),backend,tmp_path/'run.jsonl',lambda _:image_observation(path))
    dashboard=Dashboard(c.log_path,port=0);dashboard.start()
    try:
        assert not json.load(request(dashboard))['observations']
        c.acquire();c.move([5,0],'Test move');c.acquire()
        state=json.load(request(dashboard))
        assert len(state['observations'])==2 and len(state['moves'])==1
        assert state['observations'][-1]['line']['point']==[5,0]
        assert state['image'].startswith('data:image/png;base64,')
        with c.log_path.open('a') as stream:stream.write('{"event":')
        assert json.load(request(dashboard))['observations']==state['observations']
        counts=(len(backend.moves),backend.acquisitions)
        for path,method,status in [('api/status','POST',501),('../alignment.json','GET',404),('api/status?image=/etc/passwd','GET',404)]:
            with pytest.raises(HTTPError) as error:request(dashboard,path,method)
            assert error.value.code==status
        assert counts==(len(backend.moves),backend.acquisitions)
        html=request(dashboard,'').read()
        assert b'Symmetry lines and search trajectory' in html and b'Shift-drag to zoom' in html
    finally:dashboard.close()
    assert dashboard.snapshot_path.exists() and not dashboard.thread.is_alive()


@pytest.mark.parametrize('size',[200,2000])
@pytest.mark.parametrize('valid',[True,False])
def test_axis_overlay_and_arrow_at_thumbnail_scale(tmp_path,size,valid):
    path=tmp_path/'image.tiff';Image.fromarray(np.zeros((size,size),np.uint8)).save(path)
    data=Dashboard.preview(image_observation(path,size,valid))
    preview=Image.open(BytesIO(base64.b64decode(data['image'].split(',')[1])))
    assert preview.size==(min(1000,size),)*2
    scale=preview.width/size
    center=int(round((size/2+.5)*scale-.5))
    color=(0,255,255) if valid else (255,153,0)
    assert preview.getpixel((center,center))==color
    assert 'arrow points' in data['image_note']
    # The arrowhead extends to either side of the axis only toward image +y.
    tip_y=int(round(((size-1)/2+.28*size+.5)*scale-.5))
    pixels = np.asarray(preview)
    assert np.any(np.all(pixels[tip_y-12:tip_y-4, center+3:center+6] == color, axis=-1))
    assert not np.any(pixels[2*center-tip_y+4:2*center-tip_y+12, center+3:center+6])


def test_preview_crops_before_overlay(tmp_path):
    path=tmp_path/'image.tiff';Image.fromarray(np.zeros((100,120),np.uint8)).save(path)
    obs=image_observation(path)
    obs['metrics']['crop_xyxy']=[10,20,42,52]
    data=Dashboard.preview(obs)
    image=Image.open(BytesIO(base64.b64decode(data['image'].split(',')[1])))
    assert image.size==(32,32)
    assert image.getpixel((16,16))==(0,255,255)


def test_historical_images_and_snapshot_escaping(tmp_path):
    c=align_osa.Controller(settings(),Backend(),tmp_path/'run.jsonl')
    for i in (1,2,3):
        path=tmp_path/f'image-{i}.tiff'
        if i!=2:Image.fromarray(np.full((32,32),i,np.uint8)).save(path)
        obs=image_observation(path,valid=i==1)
        obs.update(acquisition=i,position=[i,0],stage='search',line=None)
        c.record('observation',observation=obs)
    c.record('fault',text="</script><script>alert('test')</script>")
    dashboard=Dashboard(c.log_path,port=0);dashboard.start()
    try:
        state=json.load(request(dashboard))
        first=json.load(request(dashboard,'api/images/1'))
        missing=json.load(request(dashboard,'api/images/2'))
        assert first['image']!=state['image']
        assert missing['image'] is None
        for route in ['api/images/0','api/images/4','api/images/1?image=/etc/passwd']:
            with pytest.raises(HTTPError):request(dashboard,route)
    finally:dashboard.close()
    html=dashboard.snapshot_path.read_text()
    assert '</script><script>' not in html
    saved=json.loads(html.split('const initialState = ',1)[1].split(';\n',1)[0])
    assert saved['observations'][0]['preview']==first
    assert saved['observations'][1]['preview']==missing


@pytest.mark.parametrize('blocked',[False,True])
def test_cli_with_and_without_dashboard(tmp_path,monkeypatch,blocked):
    outputs=[]
    for enabled in (False,True):
        folder=tmp_path/str(enabled);folder.mkdir()
        (folder/'config.json').write_text(json.dumps({'units':'um','bounds':[[-100,100],[-100,100]],'valid_axes_required':2}))
        backend=Backend()
        if blocked:
            def fail():raise RuntimeError('Synthetic acquisition failure')
            backend.acquire_image=fail
        monkeypatch.setattr(align_osa.importlib,'import_module',lambda name:type('Adapter',(),{'create':staticmethod(lambda options:backend)}))
        observations=iter([observation((0,-1)),observation((1,-1))])
        monkeypatch.setattr(align_osa,'measure',lambda *a,**k:next(observations))
        argv=['align_osa.py','--config',str(folder/'config.json'),'--adapter','synthetic:create',
              '--log',str(folder/'run.jsonl'),'--output',str(folder/'result.json')]
        if enabled:argv+=['--webui','--webui-port','0']
        monkeypatch.setattr(sys,'argv',argv)
        if blocked:
            with pytest.raises(SystemExit) as exc:align_osa.main()
            assert exc.value.code==2
        else:align_osa.main()
        result=json.loads((folder/'result.json').read_text());result.pop('log');outputs.append(result)
        assert (folder/'run.html').exists()==enabled
    assert outputs[0]==outputs[1]


def test_existing_snapshot_is_preserved(tmp_path):
    c=align_osa.Controller(settings(),Backend(),tmp_path/'run.jsonl')
    dashboard=Dashboard(c.log_path,port=0)
    dashboard.snapshot_path.write_text('existing report')
    dashboard.start();dashboard.close()
    assert dashboard.snapshot_path.read_text()=='existing report'
