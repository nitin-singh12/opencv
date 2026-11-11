"""Independent PNG encoders/decoders, preservation, malformed input and file/worker faults."""
import base64
from dataclasses import asdict,replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib
import pytest
png=pytest.importorskip('png',reason='Install png-test extra')
pytest.importorskip('oxipng',reason='Install png extra')
from PIL import Image,ImageCms,PngImagePlugin
from opencv_bv_tools.png_compression import PNGLimits
from opencv_bv_tools.png_compression import PNGCompressionError
from opencv_bv_tools.png_compression import compress_png
from opencv_bv_tools.png_compression import _run_worker
from opencv_bv_tools.png_compression import compress_file

SIG=b'\x89PNG\r\n\x1a\n'


def chunk(kind,value):
    return struct.pack('>I',len(value))+kind+value+struct.pack('>I',zlib.crc32(kind+value)&0xffffffff)


def chunks(data):
    position=8;result=[]
    while position<len(data):
        n=struct.unpack_from('>I',data,position)[0]
        result.append((data[position+4:position+8],data[position+8:position+8+n]))
        position+=12+n
    return result


def assemble(parts):return SIG+b''.join(chunk(k,v) for k,v in parts)


def pillow_png(mode='RGBA',size=(31,29),level=0,**kwargs):
    im=Image.new(mode,size)
    bands=len(im.getbands())
    im.putdata([((i%2 if mode=='1' else i%256) if bands==1 else
                 tuple((i*(17+b*31))%256 for b in range(bands))) for i in range(im.width*im.height)])
    if mode=='P':im.putpalette([(i*13)%256 for i in range(768)])
    f=BytesIO();im.save(f,format='PNG',compress_level=level,**kwargs);return f.getvalue()


def verify(source,result):
    assert result['status'] in ('optimized','unchanged'),result
    output=result['data'];assert len(output)<=len(source)
    assert result['bytes_saved']==len(source)-len(output)
    assert [(k,v) for k,v in chunks(source) if k!=b'IDAT']==[(k,v) for k,v in chunks(output) if k!=b'IDAT']
    with Image.open(BytesIO(source)) as a,Image.open(BytesIO(output)) as b:
        a.load();b.load();assert (a.mode,a.size,a.tobytes())==(b.mode,b.size,b.tobytes())
        assert a.convert('RGBA').tobytes()==b.convert('RGBA').tobytes()
    # PyPNG is independent of both Pillow and the Rust optimizer.
    aa=png.Reader(bytes=source).asDirect();bb=png.Reader(bytes=output).asDirect()
    assert aa[:2]==bb[:2] and [list(r) for r in aa[2]]==[list(r) for r in bb[2]]
    if result['status']=='unchanged':assert output==source and result['bytes_saved']==0
    return output


@pytest.mark.parametrize('mode',['1','L','LA','P','RGB','RGBA'])
def test_pillow_modes_and_non_growing_repeat(mode):
    data=pillow_png(mode);output=verify(data,compress_png(data))
    verify(output,compress_png(output))


@pytest.mark.parametrize('interlace',[False,True])
@pytest.mark.parametrize('size',[(1,1),(2,3),(3,2),(5,7),(9,9)])
@pytest.mark.parametrize('kind,depth',[('gray',1),('gray',2),('gray',4),('gray',8),
    ('palette',1),('palette',2),('palette',4),('palette',8),('rgb',8),('rgba',8),('la',8)])
def test_independent_encoder_depths_adam7_empty_passes(kind,depth,size,interlace):
    w,h=size;bands={'rgb':3,'rgba':4,'la':2}.get(kind,1)
    rows=[[(i*3+y*7)%(2**depth) for i in range(w*bands)] for y in range(h)]
    kwargs={'greyscale':kind in ('gray','la'),'alpha':kind in ('rgba','la')}
    if kind=='palette':kwargs={'palette':[(i,i*3%256,i*5%256,(255-i)%256) for i in range(2**depth)]}
    f=BytesIO();png.Writer(w,h,bitdepth=depth,interlace=interlace,compression=0,**kwargs).write(f,rows)
    verify(f.getvalue(),compress_png(f.getvalue()))


@pytest.mark.parametrize('orientation',range(1,9))
def test_orientation_and_rendering_metadata_preserved(orientation):
    info=PngImagePlugin.PngInfo();info.add_text('Description','retain this text')
    info.add_text('Compressed','compressed text '*20,zip=True)
    info.add_itxt('International','café 日本語',zip=True)
    info.add(b'gAMA',struct.pack('>I',45455));info.add(b'pHYs',struct.pack('>IIB',3000,3000,1))
    info.add(b'vpAg',b'private safe-to-copy',after_idat=True)
    exif=Image.Exif();exif[274]=orientation
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    data=pillow_png('RGBA',pnginfo=info,exif=exif,icc_profile=icc)
    output=verify(data,compress_png(data))
    with Image.open(BytesIO(output)) as im:assert im.getexif()[274]==orientation and im.info['icc_profile']==icc


def test_unused_palette_and_transparent_hidden_samples():
    data=pillow_png('P',size=(8,8),transparency=bytes(range(256)))
    verify(data,compress_png(data))
    im=Image.new('RGBA',(64,32));im.putdata([(i%256,i*13%256,i*7%256,0) for i in range(2048)])
    f=BytesIO();im.save(f,format='PNG',compress_level=0)
    verify(f.getvalue(),compress_png(f.getvalue()))


@pytest.mark.parametrize('count',[1,2,31])
def test_split_and_empty_consecutive_idat(count):
    data=pillow_png();parts=chunks(data);out=[]
    for k,v in parts:
        if k!=b'IDAT':out.append((k,v));continue
        out.append((b'IDAT',b''))
        step=max(1,len(v)//count)
        out.extend((b'IDAT',v[i:i+step]) for i in range(0,len(v),step))
        out.append((b'IDAT',b''))
    source=assemble(out);verify(source,compress_png(source))


def malformed_cases():
    valid=pillow_png('RGB');parts=chunks(valid);ihdr=parts[0][1];idat=next(v for k,v in parts if k==b'IDAT')
    return [valid[:-1],valid+b'trailing',valid[:20],b'not png',
        assemble([(b'IHDR',ihdr),(b'IHDR',ihdr),(b'IDAT',idat),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'IDAT',idat[:5]),(b'tEXt',b'key\0middle'),(b'IDAT',idat[5:]),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'IDAT',b'not zlib'),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'IDAT',idat+zlib.compress(b'extra')),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'IDAT',idat+b'extra'),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'IDAT',zlib.compress(b'x'*1000000)),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'IDAT',zlib.compress(b'\0')),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'IDAT',b''),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'IDAT',idat)]),
        assemble([(b'IDAT',idat),(b'IHDR',ihdr),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'ABCD',b'unknown'),(b'IDAT',idat),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'vpAG',b'unsafe'),(b'IDAT',idat),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'vpbg',b'reserved'),(b'IDAT',idat),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'iCCP',b'name\0\0invalid'),(b'IDAT',idat),(b'IEND',b'')]),
        assemble([(b'IHDR',ihdr),(b'iTXt',b'key\0\1\0en\0label\0invalid'),(b'IDAT',idat),(b'IEND',b'')]),
        valid[:29]+b'\0\0\0\0'+valid[33:]]


@pytest.mark.parametrize('data',malformed_cases())
def test_invalid_pngs_never_return_output(data):
    result=compress_png(data)
    assert result['status']=='failed' and result['data'] is None,result


@pytest.mark.parametrize('kind',[b'caBX',b'iDOT',b'dSIG',b'acTL',b'fcTL',b'fdAT'])
def test_signatures_offset_metadata_and_animation_rejected(kind):
    parts=chunks(pillow_png());parts.insert(1,(kind,b'12345678'))
    assert compress_png(assemble(parts))['status']=='failed'


def test_apng_and_16bit_rejected():
    a=Image.new('RGB',(10,10),'red');b=Image.new('RGB',(10,10),'blue');f=BytesIO()
    a.save(f,format='PNG',save_all=True,append_images=[b],duration=20)
    assert compress_png(f.getvalue())['status']=='failed'
    for greyscale in (False,True):
        f=BytesIO();png.Writer(2,2,greyscale=greyscale,bitdepth=16).write(f,[[0,12345]*(1 if greyscale else 3)]*2)
        assert compress_png(f.getvalue())['status']=='failed'


def test_limits_exact_and_one_over():
    data=pillow_png('RGBA',size=(31,29));n=len(chunks(data))
    verify(data,compress_png(data,limits=PNGLimits(max_input_bytes=len(data),max_pixels=899,max_dimension=31,max_chunks=n)))
    for limits,reason in [(PNGLimits(max_pixels=898),'input_pixel_limit'),(PNGLimits(max_dimension=30),'input_dimension_limit'),
                         (PNGLimits(max_chunks=n-1),'chunk_limit')]:
        r=compress_png(data,limits=limits);assert r['status']=='resource_limited' and r['stop_reason']==reason
    with pytest.raises(PNGCompressionError):compress_png(data,limits=PNGLimits(max_input_bytes=len(data)-1))
    info=PngImagePlugin.PngInfo();info.add(b'vpAg',b'abcdefg')
    metadata=pillow_png(pnginfo=info)
    verify(metadata,compress_png(metadata,limits=PNGLimits(max_metadata_bytes=7)))
    assert compress_png(metadata,limits=PNGLimits(max_metadata_bytes=6))['status']=='resource_limited'
    parts=chunks(data);parts.insert(1,(b'zTXt',b'key\0\0'+zlib.compress(b'A'*2_000_000)))
    assert compress_png(assemble(parts))['status']=='resource_limited'


@pytest.mark.parametrize('field,value',[('max_pixels',True),('max_pixels',0),('max_pixels',20_000_001),
    ('max_dimension',32769),('max_chunks',4097),('max_metadata_bytes',1.5),('max_input_bytes',-1),
    ('timeout_seconds',0),('timeout_seconds',True),('timeout_seconds',float('nan')),
    ('timeout_seconds',float('inf')),('timeout_seconds',31)])
def test_bad_limits(field,value):
    with pytest.raises(PNGCompressionError):compress_png(b'x',limits=replace(PNGLimits(),**{field:value}))


@pytest.mark.parametrize('data',[b'',bytearray(b'x'),'x',None])
def test_bad_input_type(data):
    with pytest.raises(PNGCompressionError):compress_png(data)


@pytest.mark.parametrize('kwargs',[{'limits':{}},{'cancelled':False}])
def test_bad_api_options(kwargs):
    with pytest.raises(PNGCompressionError):compress_png(b'x',**kwargs)


def test_native_dependency_version_and_bad_representation(monkeypatch):
    from opencv_bv_tools.runtime import png_worker as worker
    import oxipng
    data=pillow_png();options={'limits':asdict(PNGLimits())}
    monkeypatch.setattr(worker,'version',lambda name:'0')
    assert worker.optimize(data,options)['stop_reason']=='unsupported_optimizer_version'
    monkeypatch.setattr(worker,'version',lambda name:'9.1.1')
    monkeypatch.setattr(oxipng,'optimize_from_memory',lambda *a,**k:pillow_png('RGB'))
    guarded=worker.optimize(data,options)
    assert guarded['status']=='unchanged' and guarded['stop_reason']=='representation_preservation_guard'
    assert base64.b64decode(guarded['data_base64'])==data
    monkeypatch.setitem(sys.modules,'oxipng',None)
    assert worker.optimize(data,options)['stop_reason']=='missing_png_dependency'


def test_unchanged_is_exact_original(monkeypatch):
    import oxipng
    from opencv_bv_tools.runtime.png_worker import optimize
    data=pillow_png()
    monkeypatch.setattr(oxipng,'optimize_from_memory',lambda data,**kwargs:data)
    r=optimize(data,{'limits':asdict(PNGLimits())})
    assert r['status']=='unchanged' and r['bytes_saved']==0
    assert base64.b64decode(r['data_base64'])==data


def fake_worker(code,timeout=.3,cancelled=None,data=b'x'*100000):
    return _run_worker(data,{'limits':asdict(PNGLimits())},timeout=timeout,cancelled=cancelled,
                       command=[sys.executable,'-c',code])


@pytest.mark.parametrize('code,reason',[
    ('import time;time.sleep(10)','deadline_exceeded'),
    ("import sys,time;sys.stdout.write('{');sys.stdout.flush();time.sleep(10)",'deadline_exceeded'),
    ("import sys;sys.stdout.buffer.write(b'x'*2000000)",'worker_output_limit'),
    ("import sys;sys.stderr.buffer.write(b'x'*10000)",'worker_diagnostic_limit'),
    ('import os;os._exit(7)','worker_failed'),("print('{}')",'invalid_worker_response'),
])
def test_protocol_faults_reap_without_reader_keys(code,reason):
    import psutil
    before={p.pid for p in psutil.Process().children()}
    result=fake_worker(code)
    assert result['stop_reason']==reason and result['data'] is None and 'search_status' not in result
    assert {p.pid for p in psutil.Process().children()}==before


@pytest.mark.parametrize('code',[
    'import time;time.sleep(10)',
    "import sys,time;sys.stdin.buffer.read();sys.stdout.write('{'+(' '*100000));sys.stdout.flush();time.sleep(10)",
])
def test_cancel_input_and_output_transfer(code):
    import psutil
    before=psutil.Process().num_fds();start=time.monotonic()
    result=fake_worker(code,timeout=2,cancelled=lambda:time.monotonic()-start>.12)
    assert result['stop_reason']=='cancelled' and result['data'] is None
    assert psutil.Process().num_fds()==before


def test_pre_cancel_timeout_and_parent_validation_deadline():
    data=pillow_png()
    assert compress_png(data,cancelled=lambda:True)['stop_reason']=='cancelled'
    assert compress_png(data,limits=PNGLimits(timeout_seconds=.00001))['status']=='budget_exhausted'
    valid={'status':'unchanged','stop_reason':'test','input_bytes':len(data),'output_bytes':len(data),
           'bytes_saved':0,'data_base64':base64.b64encode(data).decode(),'image_size':[31,29],
           'output_sha256':hashlib.sha256(data).hexdigest()}
    r=fake_worker('print('+repr(json.dumps(valid))+')',data=data)
    assert r['status']=='unchanged' and r['data']==data
    valid['bytes_saved']=1
    assert fake_worker('print('+repr(json.dumps(valid))+')',data=data)['stop_reason']=='invalid_worker_response'


def test_cli_atomic_output_and_no_overwrite(tmp_path):
    data=pillow_png();source=tmp_path/'input.png';target=tmp_path/'output.png';source.write_bytes(data)
    result=compress_file(source,target);verify(data,result);assert target.read_bytes()==result['data']
    assert source.read_bytes()==data
    with pytest.raises(FileExistsError):compress_file(source,target)
    verify(data,compress_file(source,target,overwrite=True))
    run=subprocess.run([sys.executable,'-m','opencv_bv_tools.png_compression',str(source),str(tmp_path/'cli.png')],
                       capture_output=True,timeout=10)
    assert run.returncode==0 and 'data' not in json.loads(run.stdout)
    bad=subprocess.run([sys.executable,'-m','opencv_bv_tools.png_compression',str(source),str(target),'--timeout','nan'],
                       capture_output=True,timeout=5)
    assert bad.returncode==2 and json.loads(bad.stderr)['error']


@pytest.mark.parametrize('kind',['same','hardlink','symlink','fifo','directory','readonly'])
def test_file_boundaries(tmp_path,kind):
    source=tmp_path/'in.png';source.write_bytes(pillow_png());target=tmp_path/'out.png'
    if kind=='same':target=source
    elif kind=='hardlink':os.link(source,target)
    elif kind=='symlink':target.symlink_to(source)
    elif kind=='fifo':os.mkfifo(target)
    elif kind=='directory':target.mkdir()
    elif kind=='readonly':target.write_bytes(b'old');target.chmod(0o444)
    before=source.read_bytes()
    with pytest.raises((PNGCompressionError,OSError)):compress_file(source,target,overwrite=True)
    assert source.read_bytes()==before


def test_source_fifo_invalid_output_and_failed_worker_no_publish(tmp_path):
    fifo=tmp_path/'pipe';os.mkfifo(fifo)
    with pytest.raises(PNGCompressionError):compress_file(fifo,tmp_path/'out.png')
    source=tmp_path/'in.png';source.write_bytes(b'invalid')
    assert compress_file(source,tmp_path/'out.png')['status']=='failed'
    assert not (tmp_path/'out.png').exists()
    with pytest.raises(PNGCompressionError):compress_file(source,tmp_path/'out.jpg')


def test_no_clobber_race_and_failed_publish_cleanup(tmp_path,monkeypatch):
    from opencv_bv_tools import png_compression as png_cli
    source=tmp_path/'in.png';source.write_bytes(pillow_png());target=tmp_path/'out.png'
    real=png_cli.compress_png
    def raced(data,**kwargs):
        result=real(data,**kwargs);target.write_bytes(b'concurrent winner');return result
    monkeypatch.setattr(png_cli,'compress_png',raced)
    with pytest.raises(FileExistsError):compress_file(source,target)
    assert target.read_bytes()==b'concurrent winner' and not list(tmp_path.glob('.opencv-bv-*'))
    target.unlink();monkeypatch.setattr(png_cli,'compress_png',real)
    def broken(*a,**k):raise OSError('simulated publication failure')
    monkeypatch.setattr(os,'link',broken)
    with pytest.raises(OSError):compress_file(source,target)
    assert not target.exists() and not list(tmp_path.glob('.opencv-bv-*'))


def test_same_representation_changed_samples_never_published(monkeypatch):
    import oxipng
    from opencv_bv_tools.runtime.png_worker import optimize
    data=pillow_png('RGB',level=0)
    im=Image.new('RGB',(31,29),'black');b=BytesIO();im.save(b,format='PNG')
    monkeypatch.setattr(oxipng,'optimize_from_memory',lambda *a,**k:b.getvalue())
    assert optimize(data,{'limits':asdict(PNGLimits())})['stop_reason']=='optimizer_verification_failed'


def test_invalid_filter_even_with_valid_size_and_crc():
    data=pillow_png('RGB',size=(2,2));parts=chunks(data)
    damaged=assemble([(k,zlib.compress(bytes([5])+b'\0'*6+bytes([5])+b'\0'*6) if k==b'IDAT' else v) for k,v in parts])
    assert compress_png(damaged)['status']=='failed'


def test_parent_validation_overrun_discards_output(monkeypatch):
    from opencv_bv_tools import png_compression as api
    data=pillow_png();real=api._validate_response
    result={'status':'unchanged','stop_reason':'test','input_bytes':len(data),'output_bytes':len(data),
            'bytes_saved':0,'data_base64':base64.b64encode(data).decode(),'image_size':[31,29],
            'output_sha256':hashlib.sha256(data).hexdigest()}
    def slow(*args):real(*args);time.sleep(.2)
    monkeypatch.setattr(api,'_validate_response',slow)
    response=fake_worker('print('+repr(json.dumps(result))+')',data=data,timeout=.1)
    assert response['status']=='budget_exhausted' and response['data'] is None
