"""Independent PDF parsing, serialized geometry and image-to-PDF contracts."""
import base64
from dataclasses import asdict,replace
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
pytest.importorskip('pypdf')
from PIL import Image,ImageOps,PngImagePlugin
from pypdf import PdfReader
from pypdf.generic import ContentStream
from opencv_bv_tools import document_scanner as api
from opencv_bv_tools.runtime import scanner_worker as worker
from opencv_bv_tools import document_scanner as scanner_cli


def encode(im,fmt='PNG',**kwargs):
    b=BytesIO();im.save(b,format=fmt,**kwargs);return b.getvalue()


def pattern(mode='RGB',size=(31,19)):
    im=Image.new(mode,size)
    count=len(im.getbands())
    im.putdata([(i%2 if mode=='1' else i*11%256) if count==1 else tuple(i*(b*13+17)%256 for b in range(count)) for i in range(size[0]*size[1])])
    if mode=='P':im.putpalette([i*17%256 for i in range(768)])
    return im


def options(rotation=0,dpi=150,**limits):return dict(limits=asdict(api.ScanLimits(**limits)),rotation=rotation,dpi=dpi)


def expected_pixels(data):
    image=ImageOps.exif_transpose(Image.open(BytesIO(data))).convert('RGBA')
    return Image.alpha_composite(Image.new('RGBA',image.size,'white'),image).convert('RGB')


def inspect_pdf(data):
    reader=PdfReader(BytesIO(data),strict=True)
    assert not reader.is_encrypted and len(reader.pages)==1
    root=reader.trailer['/Root']
    for key in ['/OpenAction','/AA','/Names','/AcroForm','/Outlines']:assert key not in root
    page=reader.pages[0]
    for key in ['/Annots','/AA']:assert key not in page
    assert page.get('/Rotate',0)==0 and page.extract_text()==''
    operations=ContentStream(page.get_contents(),reader).operations
    assert not any(op in (b'Tj',b'TJ',b"'",b'"') for _,op in operations)
    assert len(page.images)==1
    obj=next(iter(page['/Resources']['/XObject'].values())).get_object()
    assert obj['/ColorSpace']=='/DeviceRGB' and obj['/BitsPerComponent']==8 and '/SMask' not in obj
    assert '/FlateDecode' in obj['/Filter'] and '/DCTDecode' not in obj['/Filter']
    # Independently accumulate actual serialized PDF transforms until image Do.
    stack=[];matrix=(1.,0.,0.,1.,0.,0.);image_matrix=None
    for operands,op in operations:
        if op==b'q':stack.append(matrix)
        elif op==b'Q':matrix=stack.pop()
        elif op==b'cm':
            a,b,c,d,e,f=map(float,operands);A,B,C,D,E,F=matrix
            matrix=(A*a+C*b,B*a+D*b,A*c+C*d,B*c+D*d,A*e+C*f+E,B*e+D*f+F)
        elif op==b'Do':image_matrix=matrix
    assert image_matrix is not None
    a,b,c,d,e,f=image_matrix
    corners=[(a*x+c*y+e,b*x+d*y+f) for x,y in [(0,0),(1,0),(1,1),(0,1)]]
    width,height=float(page.mediabox.width),float(page.mediabox.height)
    assert all(0<=x<=width and 0<=y<=height for x,y in corners)
    # White background explicitly fills the full written MediaBox.
    assert ([1,1,1],b'rg') in operations
    assert any(op==b're' and list(map(float,args))==[0,0,width,height] for args,op in operations)
    return reader,page,page.images[0].image,image_matrix


@pytest.mark.parametrize('orientation',range(1,9))
@pytest.mark.parametrize('rotation',[0,90,-90,180,-180])
def test_exif_then_clockwise_rotation_preserves_samples(orientation,rotation):
    exif=Image.Exif();exif[274]=orientation;exif[270]='SOURCE_METADATA_SECRET'
    data=encode(pattern(),exif=exif)
    result=api.scan_document(data,rotation=rotation)
    assert result['status']=='completed',result
    reader,page,image,matrix=inspect_pdf(result['data'])
    expected=expected_pixels(data)
    assert image.size==expected.size and image.tobytes()==expected.tobytes()
    assert image.size==tuple(result['image_size'])
    assert reader.metadata['/Creator']=='opencv-bv-tools'
    assert b'SOURCE_METADATA_SECRET' not in result['data']
    # Matrix's unit-x vector establishes clockwise direction independently.
    actual=math.degrees(math.atan2(matrix[1],matrix[0]))
    assert abs((actual+rotation+180)%360-180)<1e-5


@pytest.mark.parametrize('mode',['1','L','LA','P','RGB','RGBA'])
def test_all_modes_alpha_hidden_samples_and_metadata(mode):
    info=PngImagePlugin.PngInfo();info.add_text('Comment','DO_NOT_COPY_COMMENT');info.add_itxt('XML:com.adobe.xmp','DO_NOT_COPY_XMP')
    data=encode(pattern(mode),pnginfo=info,**({'transparency':0} if mode=='P' else {}))
    result=api.scan_document(data);_,_,image,_=inspect_pdf(result['data'])
    assert image.tobytes()==expected_pixels(data).tobytes()
    assert not image.info and image.mode=='RGB'
    assert b'DO_NOT_COPY_COMMENT' not in result['data'] and b'DO_NOT_COPY_XMP' not in result['data']


def test_fully_transparent_rgb_discarded():
    im=pattern('RGBA');im.putalpha(0)
    result=api.scan_document(encode(im));_,_,image,_=inspect_pdf(result['data'])
    assert image.getextrema()==((255,255),)*3


@pytest.mark.parametrize('fmt',['JPEG','WEBP'])
def test_decoded_input_samples_exact_and_no_original_jpeg_passthrough(fmt):
    data=encode(pattern(size=(200,100)),fmt,quality=47)
    result=api.scan_document(data);_,_,image,_=inspect_pdf(result['data'])
    assert image.tobytes()==expected_pixels(data).tobytes()


@pytest.mark.parametrize('size',[(1,1),(1,16384),(16384,1),(101,103),(14399,1)])
@pytest.mark.parametrize('angle',[-180,-90,-89.99999,-45,-.000001,0,.000001,7.5,45,89.99999,90,180])
def test_serialized_corners_extreme_aspects_near_right_and_tiny_angles(size,angle):
    data=encode(Image.new('RGB',size,'black'))
    result=worker.process(data,options(rotation=angle,dpi=150))
    assert result['status']=='completed',result
    _,page,_,_=inspect_pdf(base64.b64decode(result['data_base64']))
    assert [float(page.mediabox.width),float(page.mediabox.height)]==result['page_size_points']


@pytest.mark.parametrize('dpi',[72,150,300,600])
def test_dpi_sets_physical_size_and_ignores_input_dpi(dpi):
    result=api.scan_document(encode(pattern(size=(400,600)),dpi=(9999,9999)),dpi=dpi)
    _,page,_,_=inspect_pdf(result['data'])
    assert .09<=float(page.mediabox.width)-400*72/dpi<=.13
    assert .09<=float(page.mediabox.height)-600*72/dpi<=.13


def test_page_size_ceiling_rejects_without_crop():
    for size,expected in [((14399,1),'completed'),((14400,1),'resource_limited')]:
        result=api.scan_document(encode(Image.new('RGB',size)),dpi=72)
        assert result['status']==expected,result
        if expected!='completed':assert result['data'] is None and result['stop_reason']=='pdf_page_size_limit'


def test_deterministic_repeat_and_equivalent_half_turns():
    data=encode(pattern())
    first=api.scan_document(data,rotation=180);second=api.scan_document(data,rotation=-180)
    assert first['data']==second['data']==api.scan_document(data,rotation=180)['data']


@pytest.mark.parametrize('angle',[True,None,'90',float('nan'),float('inf'),float('-inf'),180.0001,-180.0001,270])
def test_invalid_rotation(angle):
    with pytest.raises(api.DocumentScanError):api.scan_document(b'input',rotation=angle)


@pytest.mark.parametrize('dpi',[True,71,601,150.0,None,'150'])
def test_invalid_dpi(dpi):
    with pytest.raises(api.DocumentScanError):api.scan_document(b'input',dpi=dpi)


@pytest.mark.parametrize('field',list(asdict(api.ScanLimits())))
@pytest.mark.parametrize('value',[True,0,-1,'1',float('nan'),float('inf')])
def test_invalid_limits(field,value):
    with pytest.raises(api.DocumentScanError):api.scan_document(b'x',limits=replace(api.ScanLimits(),**{field:value}))


@pytest.mark.parametrize('data',[None,b'','input',bytearray(b'x')])
def test_invalid_input_types(data):
    with pytest.raises(api.DocumentScanError):api.scan_document(data)


@pytest.mark.parametrize('kwargs',[{'limits':{}},{'cancelled':True},{'limits':api.ScanLimits(max_input_bytes=1)},
 {'limits':api.ScanLimits(max_pixels=20_000_001)},{'limits':api.ScanLimits(timeout_seconds=31)}])
def test_invalid_options(kwargs):
    with pytest.raises(api.DocumentScanError):api.scan_document(b'input',**kwargs)


@pytest.mark.parametrize('limits',[api.ScanLimits(max_pixels=10),api.ScanLimits(max_dimension=10),api.ScanLimits(max_output_bytes=100)])
def test_resource_limit_discards_output(limits):
    result=api.scan_document(encode(pattern()),limits=limits)
    assert result['status']=='resource_limited' and result['data'] is None


@pytest.mark.parametrize('data',[b'%PDF-1.4\n%%EOF',b'not image',b'\x89PNG\r\n\x1a\n',b'RIFF\0\0\0\0WEBP',b'\xff\xd8\xff'])
def test_invalid_or_pdf_input_rejected(data):
    result=api.scan_document(data);assert result['status']=='failed' and result['data'] is None


def test_high_depth_profiles_animation_and_unsupported_color():
    inputs=[encode(Image.new('I;16',(10,10))),encode(Image.new('CMYK',(10,10)),'JPEG'),
        encode(pattern(),icc_profile=b'fake-profile'),encode(pattern(),'GIF')]
    for fmt in ['PNG','WEBP']:
        inputs.append(encode(Image.new('RGB',(10,10),'red'),fmt,save_all=True,
                             append_images=[Image.new('RGB',(10,10),'blue')],duration=10))
    for data in inputs:
        result=api.scan_document(data);assert result['status']=='failed' and result['data'] is None
    info=PngImagePlugin.PngInfo();info.add(b'gAMA',(45455).to_bytes(4,'big'))
    assert api.scan_document(encode(pattern(),pnginfo=info))['stop_reason']=='color_profile_not_supported'


def test_creation_faults(monkeypatch):
    data=encode(pattern())
    monkeypatch.setattr(worker,'create_pdf',lambda *a:(_ for _ in ()).throw(RuntimeError()))
    assert worker.process(data,options())['stop_reason']=='pdf_creation_error'


@pytest.mark.parametrize('mutation',[{'status':'bad'},{'capabilities':'automatic scanner'},{'page_count':2},
 {'page_size_points':[15000,1]},{'page_size_points':[float('nan'),1]},{'image_size':[0,1]},
 {'output_bytes':True},{'data_base64':'x'},{'output_sha256':'0'*64},{'status':'failed'}])
def test_parent_rejects_bad_protocol(mutation):
    result=worker.process(encode(pattern()),options());result.update(mutation)
    with pytest.raises((ValueError,KeyError,TypeError)):api._validate_response(result,options()['limits'])


def test_scanner_supervision_cancellation_timeout_and_protocol():
    assert api.scan_document(b'x',cancelled=lambda:True)['stop_reason']=='cancelled'
    for command,reason in [('import time;time.sleep(3)','deadline_exceeded'),('print("{}")','invalid_worker_response')]:
        result=api._run_worker(b'x',options(),timeout=.25,command=[sys.executable,'-c',command])
        assert result['stop_reason']==reason and result['data'] is None


@pytest.mark.parametrize('status',['failed','resource_limited','budget_exhausted'])
def test_failure_preserves_existing_output(monkeypatch,tmp_path,status):
    src=tmp_path/'in.png';src.write_bytes(encode(pattern()));out=tmp_path/'out.pdf';out.write_bytes(b'old')
    result=api._failure('test');result['status']=status
    monkeypatch.setattr(scanner_cli,'scan_document',lambda *a,**k:result)
    assert scanner_cli.scan_file(src,out,overwrite=True)['status']==status and out.read_bytes()==b'old'


def test_final_cancellation_and_atomic_race(monkeypatch,tmp_path):
    src=tmp_path/'in.png';src.write_bytes(encode(pattern()));out=tmp_path/'out.pdf';out.write_bytes(b'old')
    monkeypatch.setattr(scanner_cli,'scan_document',lambda *a,**k:{'status':'completed','data':b'new'})
    calls=[]
    def cancelled():calls.append(1);return len(calls)==2
    assert scanner_cli.scan_file(src,out,overwrite=True,cancelled=cancelled)['stop_reason']=='cancelled'
    assert out.read_bytes()==b'old' and not list(tmp_path.glob('.opencv-bv-*'))
    out.unlink()
    def race(*a,**k):out.write_bytes(b'winner');return {'status':'completed','data':b'new'}
    monkeypatch.setattr(scanner_cli,'scan_document',race)
    with pytest.raises(FileExistsError):scanner_cli.scan_file(src,out)
    assert out.read_bytes()==b'winner' and not list(tmp_path.glob('.opencv-bv-*'))


@pytest.mark.parametrize('kind',['same','hardlink','symlink','fifo','directory','readonly','extension'])
def test_output_file_policy(tmp_path,kind):
    src=tmp_path/'input.pdf';src.write_bytes(encode(pattern()));out=tmp_path/'out.pdf'
    if kind=='same':out=src
    elif kind=='hardlink':os.link(src,out)
    elif kind=='symlink':out.symlink_to(src)
    elif kind=='fifo':os.mkfifo(out)
    elif kind=='directory':out.mkdir()
    elif kind=='readonly':out.write_bytes(b'old');out.chmod(0o444)
    else:out=tmp_path/'out.png'
    with pytest.raises((api.DocumentScanError,OSError)):scanner_cli.scan_file(src,out,overwrite=True)


def test_cli_real_success_error_and_untouched_input(tmp_path):
    src=tmp_path/'private-name.png';data=encode(pattern());src.write_bytes(data);out=tmp_path/'out.pdf'
    cmd=[sys.executable,'-m','opencv_bv_tools.document_scanner',str(src),str(out),'--rotation','-7.5']
    run=subprocess.run(cmd,capture_output=True,text=True);assert run.returncode==0,run.stderr
    summary=json.loads(run.stdout);assert 'data' not in summary and summary['output']==str(out)
    inspect_pdf(out.read_bytes());assert src.read_bytes()==data and b'private-name' not in out.read_bytes()
    previous=out.read_bytes();src.write_bytes(b'bad')
    run=subprocess.run(cmd+['--overwrite'],capture_output=True,text=True)
    assert run.returncode==2 and 'output' not in json.loads(run.stdout) and out.read_bytes()==previous


def test_cli_invalid_options_and_non_regular_input(tmp_path):
    src=tmp_path/'fifo';os.mkfifo(src)
    with pytest.raises(api.DocumentScanError):scanner_cli.scan_file(src,tmp_path/'out.pdf')
    for kwargs in [{'limits':{}},{'overwrite':1},{'rotation':270}]:
        with pytest.raises(api.DocumentScanError):scanner_cli.scan_file(src,tmp_path/'out.pdf',**kwargs)
