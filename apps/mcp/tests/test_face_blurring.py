"""Face geometry, image semantics, real inference and failure-safe publication."""
import base64
from dataclasses import asdict, replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
np=pytest.importorskip('numpy')
cv=pytest.importorskip('cv2')
from PIL import Image, ImageOps, PngImagePlugin
from opencv_bv_tools.runtime import face_worker as worker
from opencv_bv_tools import face_blurring as api
from opencv_bv_tools import face_blurring as face_cli

FIXTURES=Path(__file__).parent/'fixtures'/'face'


def encode(im, fmt='PNG', **kwargs):
    b=BytesIO(); im.save(b,format=fmt,**kwargs); return b.getvalue()


def photo(name='grace_hopper.jpg'):
    with Image.open(FIXTURES/name) as im:
        im=im.convert('RGB');im.info.clear();return im


def options(**kwargs):return {'limits':asdict(api.FaceLimits(**kwargs))}


def fake_success(monkeypatch):
    monkeypatch.setattr(worker,'detect_regions',lambda im,*args:([[0,0,im.shape[1]//2,im.shape[0]//2]],1))


@pytest.mark.parametrize('name',['astronaut.png','camera.png','grace_hopper.jpg'])
@pytest.mark.parametrize('fmt',['PNG','JPEG','WEBP'])
def test_actual_cpu_inference_and_metadata_free_output(name,fmt):
    data=encode(photo(name),fmt)
    result=api.blur_faces(data)
    assert result['status']=='blurred',result
    assert result['qualification']==api.QUALIFICATION
    assert result['backend']['target']=='CPU' and result['backend']['model_sha256']==api.MODEL_SHA256
    im=Image.open(BytesIO(result['data']));im.load()
    assert im.format=='PNG' and im.mode=='RGB' and im.size==photo(name).size
    assert not im.info and not im.getexif()
    assert result['output_sha256']==hashlib.sha256(result['data']).hexdigest()


@pytest.mark.parametrize('orientation',range(1,9))
def test_exif_orientation_is_applied_before_detection(monkeypatch,orientation):
    fake_success(monkeypatch)
    im=Image.fromarray(np.random.default_rng(20261103).integers(0,256,(23,31,3),dtype=np.uint8))
    exif=Image.Exif();exif[274]=orientation;exif[270]='private description'
    data=encode(im,exif=exif)
    expected=ImageOps.exif_transpose(Image.open(BytesIO(data))).convert('RGB')
    seen=[]
    def detect(image,*args):seen.append(image.copy());return [[0,0,4,4]],1
    monkeypatch.setattr(worker,'detect_regions',detect)
    result=worker.process(data,options())
    assert seen[0].shape[:2]==(expected.height,expected.width) and seen[0].tobytes()==expected.tobytes()
    assert not Image.open(BytesIO(base64.b64decode(result['data_base64']))).getexif()


@pytest.mark.parametrize('mode',['1','L','LA','P','RGB','RGBA'])
def test_modes_transparency_and_metadata_removed(monkeypatch,mode):
    fake_success(monkeypatch)
    im=Image.new(mode,(31,29))
    if mode=='P':im.putpalette([i%256 for i in range(768)])
    info=PngImagePlugin.PngInfo();info.add_text('private','secret');info.add_itxt('more','hidden thumbnail')
    data=encode(im,pnginfo=info,**({'transparency':0} if mode=='P' else {}))
    normalized,_=worker.normalize(data,options()['limits'])
    expected=Image.alpha_composite(Image.new('RGBA',im.size,'white'),Image.open(BytesIO(data)).convert('RGBA')).convert('RGB')
    assert normalized.tobytes()==expected.tobytes()
    result=worker.process(data,options());out=Image.open(BytesIO(base64.b64decode(result['data_base64'])))
    assert out.mode=='RGB' and not out.info and b'secret' not in base64.b64decode(result['data_base64'])


def test_hidden_transparent_face_removed():
    im=photo().convert('RGBA');im.putalpha(0)
    result=api.blur_faces(encode(im))
    assert result['status']=='no_faces' and result['data'] is None
    normalized,_=worker.normalize(encode(im),options()['limits'])
    assert (normalized==255).all()


@pytest.mark.parametrize('kind',list(worker.COLOR_CHUNKS))
def test_png_color_declarations_rejected(kind):
    import struct,zlib
    data=encode(Image.new('RGB',(16,16)))
    value={b'gAMA':struct.pack('>I',45455),b'cHRM':bytes(32),b'iCCP':b'profile\0\0'+zlib.compress(b'ICC')}.get(kind,bytes(24))
    chunk=struct.pack('>I',len(value))+kind+value+struct.pack('>I',zlib.crc32(kind+value)&0xffffffff)
    r=api.blur_faces(data[:33]+chunk+data[33:])
    assert r['status']=='failed' and r['data'] is None


def test_profile_original_rejected_and_srgb_declaration_allowed():
    assert api.blur_faces((FIXTURES/'astronaut.png').read_bytes())['stop_reason']=='color_profile_not_supported'
    info=PngImagePlugin.PngInfo();info.add(b'sRGB',b'\0')
    assert api.blur_faces(encode(photo(),pnginfo=info))['status']=='blurred'


@pytest.mark.parametrize('fmt',['PNG','WEBP'])
def test_multiframe_rejected(fmt):
    a=Image.new('RGB',(20,20),'red');b=Image.new('RGB',(20,20),'blue')
    r=api.blur_faces(encode(a,fmt,save_all=True,append_images=[b],duration=20))
    assert r['status']=='failed' and r['data'] is None


@pytest.mark.parametrize('data',[b'not image',b'\x89PNG\r\n\x1a\n',b'\xff\xd8\xff',b'RIFF\0\0\0\0WEBP'])
def test_malformed_input_no_output(data):
    r=api.blur_faces(data);assert r['status']=='failed' and r['data'] is None


def test_high_depth_and_unsupported_mode():
    for data in [encode(Image.new('I;16',(2,2))),encode(Image.new('CMYK',(20,20)),'JPEG'),encode(photo(),'GIF')]:
        r=api.blur_faces(data);assert r['status']=='failed' and r['data'] is None


@pytest.mark.parametrize('size',[(31,17),(100,201),(1,57),(57,1)])
@pytest.mark.parametrize('rotation',range(4))
def test_inverse_geometry_against_rotated_raster(size,rotation):
    w,h=size;mask=Image.new('L',size)
    box=(w//4,h//4,max(w//4+1,3*w//4),max(h//4+1,3*h//4))
    mask.paste(255,box)
    rotated=mask.rotate(90*rotation,expand=True)
    rb=rotated.getbbox();rw,rh=rotated.size
    resized=(rw*2,rh*3)
    mapped=worker.inverse_box((rb[0]*2,rb[1]*3,rb[2]*2,rb[3]*3),size,rotation,resized)
    assert tuple(mapped)==box


@pytest.mark.parametrize('box',[(-1.2,-3.1,12.7,10.1),(0,0,1,1),(90,90,120,150),(20.1,25.2,42.8,67.9)])
def test_expansion_covers_fractional_box_and_clips(box):
    b=worker.expand_box(box,(100,100))
    assert b[0]<=max(0,box[0]) and b[1]<=max(0,box[1])
    assert b[2]>=min(100,box[2]) and b[3]>=min(100,box[3])
    assert all(0<=v<=100 for v in b)


def test_transitive_merging_covers_all_boxes_independent_of_order():
    import itertools
    boxes=[[0,0,5,5],[10,0,15,5],[5,0,10,5],[40,20,41,21]]
    for perm in itertools.permutations(boxes):
        regions=worker.merge_regions(perm)
        assert regions==[[0,0,15,5],[40,20,41,21]]
        for b in boxes:assert any(r[0]<=b[0] and r[1]<=b[1] and r[2]>=b[2] and r[3]>=b[3] for r in regions)
    # Later-grown rectangle reaches a component already emitted.
    boxes=[[0,0,5,5],[3,10,12,15],[10,3,15,12]]
    assert worker.merge_regions(boxes)==[[0,0,15,15]]


def test_blur_attenuates_every_region_and_preserves_outside():
    data=np.random.default_rng(17).integers(0,256,(100,120,3),dtype=np.uint8)
    regions=[[3,5,40,50],[60,50,120,100]]
    out=np.array(worker.apply_blur(data,regions));mask=np.zeros(data.shape[:2],bool)
    for x0,y0,x1,y1 in regions:
        mask[y0:y1,x0:x1]=True
        before=data[y0:y1,x0:x1].astype(float);after=out[y0:y1,x0:x1].astype(float)
        assert after.var()<before.var()*0.1
        assert not np.array_equal(before,after)
    assert np.array_equal(data[~mask],out[~mask])


@pytest.mark.parametrize('field',list(asdict(api.FaceLimits())))
@pytest.mark.parametrize('value',[0,-1,True,'1',float('nan'),float('inf')])
def test_invalid_limit_values(field,value):
    with pytest.raises(api.FaceBlurError):api.blur_faces(b'x',limits=replace(api.FaceLimits(),**{field:value}))


@pytest.mark.parametrize('kwargs',[{'limits':{}},{'cancelled':True},{'limits':api.FaceLimits(max_pixels=20_000_001)},
    {'limits':api.FaceLimits(timeout_seconds=31)},{'limits':api.FaceLimits(max_input_bytes=1)}])
def test_invalid_caller_options(kwargs):
    with pytest.raises(api.FaceBlurError):api.blur_faces(b'123',**kwargs)


@pytest.mark.parametrize('data',[b'',bytearray(b'x'),'x',None])
def test_invalid_bytes(data):
    with pytest.raises(api.FaceBlurError):api.blur_faces(data)


@pytest.mark.parametrize('limits',[api.FaceLimits(max_pixels=200),api.FaceLimits(max_dimension=10),api.FaceLimits(max_output_bytes=100)])
def test_resource_limits_no_output(limits):
    r=api.blur_faces(encode(photo()),limits=limits)
    assert r['status']=='resource_limited' and r['data'] is None


def test_region_cap_and_dependency_faults(monkeypatch):
    data=encode(photo())
    monkeypatch.setattr(worker,'detect_regions',lambda *a:([[0,0,5,5],[10,10,15,15]],2))
    assert worker.process(data,options(max_regions=1))['stop_reason']=='region_limit'
    monkeypatch.setattr(worker,'model_bytes',lambda: (_ for _ in ()).throw(ValueError()))
    assert worker.process(data,options())['stop_reason']=='missing_or_invalid_model'
    monkeypatch.setattr(worker,'version',lambda name:'unexpected')
    assert worker.process(data,options())['stop_reason']=='unsupported_detector_version'


def test_source_build_requires_explicit_matching_prefix(monkeypatch, tmp_path):
    from types import SimpleNamespace

    def absent(name):
        raise worker.PackageNotFoundError(name)

    monkeypatch.setattr(worker, 'version', absent)
    candidate = SimpleNamespace(__file__=str(tmp_path / 'cv2/__init__.py'), __version__='5.0.0-dev')
    monkeypatch.delenv('OPENCV_BV_SOURCE_BUILD', raising=False)
    assert worker.opencv_version(candidate) is None
    monkeypatch.setenv('OPENCV_BV_SOURCE_BUILD', str(tmp_path))
    assert worker.opencv_version(candidate) == '5.0.0-dev (source build)'
    candidate.__file__ = str(tmp_path.parent / 'other/cv2/__init__.py')
    assert worker.opencv_version(candidate) is None
    candidate.__file__ = str(tmp_path / 'cv2/__init__.py')
    candidate.__version__ = '4.12.0'
    assert worker.opencv_version(candidate) is None
    candidate.__version__ = '5.0.0-dev'
    monkeypatch.setattr(worker, 'version', lambda name: 'unexpected')
    assert worker.opencv_version(candidate) is None


def test_bundled_model_integrity_and_tamper(monkeypatch):
    assert len(worker.model_bytes())==229738
    monkeypatch.setattr(worker,'MODEL_SHA256','0'*64)
    with pytest.raises(ValueError,match='model_hash'):worker.model_bytes()


class FakeDetector:
    def __init__(self,found):self.found=found;self.calls=[]
    def setInputSize(self,size):self.calls.append(size)
    def detect(self,image):return None,self.found


@pytest.mark.parametrize('bad',[np.zeros((1,14)),np.full((1,15),np.nan),np.zeros((1,15))])
def test_invalid_native_boxes(monkeypatch,bad):
    monkeypatch.setattr(cv.FaceDetectorYN,'create',lambda *a:FakeDetector(bad))
    with pytest.raises(ValueError):worker.detect_regions(np.zeros((100,100,3),np.uint8),cv,np,b'ignored')


def test_candidate_cap_without_partial_result(monkeypatch):
    found=np.zeros((513,15));found[:,2:4]=20;found[:,14]=0.9
    monkeypatch.setattr(cv.FaceDetectorYN,'create',lambda *a:FakeDetector(found))
    with pytest.raises(worker.FaceResourceError):worker.detect_regions(np.asarray(photo()),cv,np,b'x')


def test_topk_exceeds_all_anchors_and_view_size_is_bounded(monkeypatch):
    calls=[];det=FakeDetector(None)
    def create(*args):calls.append(args);return det
    monkeypatch.setattr(cv.FaceDetectorYN,'create',create)
    assert worker.detect_regions(np.zeros((1,16384,3),np.uint8),cv,np,b'x')==([],0)
    assert calls[0][6]>sum((640//stride)**2 for stride in (8,16,32))
    assert det.calls==[(320,320),(640,640)]


@pytest.mark.parametrize('program,reason',[
    ('import time;time.sleep(5)','deadline_exceeded'),
    ('import sys;sys.stdout.write("x"*100000)','worker_output_limit'),
    ('import sys;sys.stderr.write("x"*10000)','worker_diagnostic_limit'),
    ('print("{}");','invalid_worker_response'),
    ('raise SystemExit(7)','worker_failed')])
def test_supervisor_faults(program,reason):
    r=api._run_worker(b'input',options(max_output_bytes=100),timeout=.25,command=[sys.executable,'-c',program])
    assert r['stop_reason']==reason and r['data'] is None


def test_cancelled_before_and_during_worker():
    assert api.blur_faces(b'x',cancelled=lambda:True)['stop_reason']=='cancelled'
    start=time.monotonic()
    r=api._run_worker(b'x',options(),timeout=2,cancelled=lambda:time.monotonic()-start>.1,
        command=[sys.executable,'-c','import time;time.sleep(5)'])
    assert r['stop_reason']=='cancelled' and time.monotonic()-start<1


@pytest.mark.parametrize('mutation',[
    {'status':'invalid'},{'qualification':'safe'},{'regions':[]},{'regions':[[0,0,999999,1]]},
    {'regions':[[True,0,2,2]]},{'output_bytes':True},{'image_size':[1,1]},
    {'output_sha256':'0'*64},{'data_base64':'x'},{'status':'no_faces'}])
def test_parent_rejects_invalid_response(monkeypatch,mutation):
    fake_success(monkeypatch);r=worker.process(encode(photo()),options());r.update(mutation)
    with pytest.raises((ValueError,KeyError,TypeError)):api._validate_response(r,options()['limits'])


@pytest.mark.parametrize('status',['failed','resource_limited','budget_exhausted','no_faces'])
def test_non_success_preserves_existing_destination(monkeypatch,tmp_path,status):
    src=tmp_path/'in.jpg';src.write_bytes(encode(photo(),'JPEG'));out=tmp_path/'out.png';out.write_bytes(b'existing')
    result=api._failure('test');result['status']=status
    monkeypatch.setattr(face_cli,'blur_faces',lambda *a,**k:result)
    assert face_cli.blur_file(src,out,overwrite=True)['status']==status
    assert out.read_bytes()==b'existing' and src.read_bytes()==encode(photo(),'JPEG')


def test_cancel_before_publish_and_racing_destination(monkeypatch,tmp_path):
    src=tmp_path/'in.png';src.write_bytes(encode(photo()));out=tmp_path/'out.png'
    def result(*a,**k):return {'status':'blurred','data':b'new'}
    monkeypatch.setattr(face_cli,'blur_faces',result)
    assert face_cli.blur_file(src,out,cancelled=lambda:True)['stop_reason']=='cancelled'
    assert not out.exists()
    def race(*a,**k):out.write_bytes(b'winner');return result()
    monkeypatch.setattr(face_cli,'blur_faces',race)
    with pytest.raises(FileExistsError):face_cli.blur_file(src,out)
    assert out.read_bytes()==b'winner' and not list(tmp_path.glob('.opencv-bv-*'))


@pytest.mark.parametrize('kind',['same','hardlink','symlink','directory','fifo','readonly','extension'])
def test_destination_guards(tmp_path,kind):
    src=tmp_path/'in.png';src.write_bytes(encode(photo()));out=tmp_path/'out.png'
    if kind=='same':out=src
    elif kind=='hardlink':os.link(src,out)
    elif kind=='symlink':out.symlink_to(src)
    elif kind=='directory':out.mkdir()
    elif kind=='fifo':
        if not hasattr(os, 'mkfifo'): pytest.skip('POSIX FIFO path contract')
        os.mkfifo(out)
    elif kind=='readonly':out.write_bytes(b'old');out.chmod(0o444)
    else:out=tmp_path/'out.jpg'
    with pytest.raises((api.FaceBlurError,OSError)):face_cli.blur_file(src,out,overwrite=True)


def test_real_cli_success_and_no_faces_does_not_claim_old_output(tmp_path):
    src=tmp_path/'in.jpg';src.write_bytes((FIXTURES/'grace_hopper.jpg').read_bytes());out=tmp_path/'out.png'
    cmd=[sys.executable,'-m','opencv_bv_tools.face_blurring',str(src),str(out)]
    run=subprocess.run(cmd,capture_output=True,text=True);assert run.returncode==0,run.stderr
    summary=json.loads(run.stdout);assert 'data' not in summary and summary['output']==str(out)
    original=out.read_bytes();src.write_bytes(encode(Image.new('RGB',(100,100),'white')))
    run=subprocess.run(cmd+['--overwrite'],capture_output=True,text=True)
    assert run.returncode==2 and 'output' not in json.loads(run.stdout) and out.read_bytes()==original


@pytest.mark.parametrize('kind', ['directory', 'fifo'])
def test_cli_rejects_non_regular_input_and_invalid_options(tmp_path, kind):
    src=tmp_path/kind
    if kind == 'fifo':
        if not hasattr(os, 'mkfifo'): pytest.skip('POSIX FIFO path contract')
        os.mkfifo(src)
    else:
        src.mkdir()
    with pytest.raises((api.FaceBlurError, OSError)):face_cli.blur_file(src,tmp_path/'out.png')
    assert not (tmp_path/'out.png').exists()
    for kwargs in [{'overwrite':1},{'limits':{}},{'limits':api.FaceLimits(max_pixels=0)}]:
        with pytest.raises(api.FaceBlurError):face_cli.blur_file(src,tmp_path/'out.png',**kwargs)


def test_cancellation_after_temporary_write_preserves_old_output(monkeypatch,tmp_path):
    src=tmp_path/'in.png';src.write_bytes(b'input');out=tmp_path/'out.png';out.write_bytes(b'old')
    monkeypatch.setattr(face_cli,'blur_faces',lambda *a,**k:{'status':'blurred','data':b'new'})
    calls=[]
    def cancel():
        calls.append(1)
        return len(calls)==2
    result=face_cli.blur_file(src,out,overwrite=True,cancelled=cancel)
    assert result['stop_reason']=='cancelled' and out.read_bytes()==b'old'
    assert not list(tmp_path.glob('.opencv-bv-*'))


def test_native_exception_and_missing_dependency(monkeypatch):
    import builtins
    real_import=builtins.__import__
    def blocked(name,*a,**k):
        if name=='cv2':raise ImportError('test')
        return real_import(name,*a,**k)
    data=encode(photo());monkeypatch.setattr(builtins,'__import__',blocked)
    assert worker.process(data,options())['stop_reason']=='missing_face_dependency'
    monkeypatch.setattr(builtins,'__import__',real_import)
    monkeypatch.setattr(worker,'detect_regions',lambda *a:(_ for _ in ()).throw(RuntimeError()))
    result=worker.process(data,options())
    assert result['stop_reason']=='detector_or_processing_error' and result['data'] is None


def test_post_validation_deadline_discards_output(monkeypatch):
    original=api._validate_response
    def slow(*args):original(*args);time.sleep(.1)
    monkeypatch.setattr(api,'_validate_response',slow)
    r=api._run_worker(b'x',options(),timeout=.07,
        command=[sys.executable,'-c','import json;print(json.dumps('+repr(api._failure('test'))+'))'])
    assert r['status']=='budget_exhausted' and r['data'] is None


def test_child_reaped_after_timeout(tmp_path):
    pidfile=tmp_path/'pid'
    command=[sys.executable,'-c',f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));time.sleep(5)']
    result=api._run_worker(b'x',options(),timeout=.2,command=command)
    assert result['stop_reason']=='deadline_exceeded'
    import psutil
    assert not psutil.pid_exists(int(pidfile.read_text()))
