"""Independent encoders, payload contracts, input policy and process supervision."""
import base64
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import pytest
pytest.importorskip('barcode',reason='Install barcode-test extra')
from PIL import Image, ImageDraw, ImageOps
from opencv_bv_tools.barcode_reader import BarcodeLimits
from opencv_bv_tools.barcode_reader import BarcodeReadError
from opencv_bv_tools.barcode_reader import DEFAULT_FORMATS
from opencv_bv_tools.barcode_reader import read_barcodes
from opencv_bv_tools.barcode_reader import _run_worker
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from barcode_cases import CASES, FIXTURES, encoded, make_code, invalid_checksum


def payloads(result):
    assert result['search_status']=='completed',result
    return [base64.b64decode(s['payload_base64']) for s in result['symbols']]


@pytest.mark.parametrize('format,value,expected',CASES)
@pytest.mark.parametrize('transform',['clean','90','180','270','invert','contrast','jpeg','webp'])
def test_independent_formats_and_transforms(format,value,expected,transform):
    im=make_code(format,value)
    if transform.isdigit(): im=im.rotate(int(transform),expand=True)
    elif transform=='invert': im=ImageOps.invert(im)
    elif transform=='contrast': im=im.point(lambda p:80+int(p*.4))
    data=encoded(im,transform.upper() if transform in ('jpeg','webp') else 'PNG')
    result=read_barcodes(data,formats=[format],itf_lengths=[len(expected)] if format=='ITF' else None)
    assert payloads(result)==[expected]
    assert result['symbols'][0]['format']==format


def test_default_excludes_optional_and_all_2d():
    import segno
    from io import BytesIO
    for name,value in [('Code39','ABC123'),('ITF','00123456'),('Code39Extended','abc')]:
        assert payloads(read_barcodes(encoded(make_code(name,value))))==[]
    qr=BytesIO();segno.make('ignore-QR',micro=False).save(qr,kind='png',scale=5)
    assert payloads(read_barcodes(qr.getvalue()))==[]
    assert payloads(read_barcodes((FIXTURES/'datamatrix.png').read_bytes()))==[]
    tile=make_code('Code128','KEEP');canvas=Image.new('RGB',(1000,600),'white')
    canvas.paste(tile,(20,20));canvas.paste(Image.open(BytesIO(qr.getvalue())),(30,300))
    canvas.paste(Image.open(FIXTURES/'datamatrix.png'),(500,300))
    assert payloads(read_barcodes(encoded(canvas)))==[b'KEEP']


@pytest.mark.parametrize('formats,expected_format',[
    (['EAN13'],'EAN13'),(['UPCA'],'UPCA'),(['EAN13','UPCA'],'EAN13'),
])
def test_upca_ean_equivalence_and_leading_zeros(formats,expected_format):
    result=read_barcodes(encoded(make_code('UPCA','01234567890')),formats=formats)
    assert payloads(result)==[b'0012345678905']
    assert result['symbols'][0]['format']==expected_format
    assert 'normalize to EAN-13' in result['payload_semantics']


@pytest.mark.parametrize('number',['0123450','0123451','0123452','0123453','0123454',
                                   '0123455','0123456','0123457','0123458','0123459','1123456'])
def test_upce_expansion_and_original_compressed_value(number):
    # Independent expansion of the encoder input and independent UPC-A checksum.
    import barcode
    ns,a,b,c,d,e,f=number
    if f in '012': expanded=ns+a+b+f+'0000'+c+d+e
    elif f=='3': expanded=ns+a+b+c+'00000'+d+e
    elif f=='4': expanded=ns+a+b+c+d+'00000'+e
    else: expanded=ns+a+b+c+d+e+'0000'+f
    full=barcode.get('upca',expanded).get_fullcode()
    result=read_barcodes(encoded(make_code('UPCE',number)),formats=['UPCE'])
    assert payloads(result)==[('0'+full).encode()]
    assert result['symbols'][0]['upc_e_compressed']==number+full[-1]
    assert payloads(read_barcodes(encoded(make_code('UPCE',number)),formats=['UPCA','EAN13']))==[]


def test_code39_checksum_and_interpretation_are_explicit():
    plain=make_code('Code39','ABC123');check=make_code('Code39','ABC123',add_checksum=True)
    assert payloads(read_barcodes(encoded(plain),formats=['Code39']))==[b'ABC123']
    checked=read_barcodes(encoded(check),formats=['Code39'])
    assert payloads(checked)==[b'ABC123$'] # Optional Mod43 is retained, not stripped.
    assert checked['symbols'][0]['checksum_policy']=='optional_not_required'
    ext=(FIXTURES/'code39ext.png').read_bytes()
    assert payloads(read_barcodes(ext,formats=['Code39']))==[b'+A+B+C']
    assert payloads(read_barcodes(ext,formats=['Code39Extended']))==[b'abc']


@pytest.mark.parametrize('value',['00','00123456','00123456789012','00123456789013'])
def test_itf_has_no_required_gtin_checksum(value):
    result=read_barcodes(encoded(make_code('ITF',value)),formats=['ITF'],itf_lengths=[len(value)])
    assert payloads(result)==[value.encode()]
    assert result['symbols'][0]['checksum_policy']=='optional_not_required'


@pytest.mark.parametrize('format',['EAN13','EAN8','UPCA','Code128'])
def test_bad_mandatory_checksum_never_returns_payload(format):
    result=read_barcodes(encoded(invalid_checksum(format)),formats=[format])
    assert payloads(result)==[]


def test_addon_ignored_and_gs1_control_bytes_preserved():
    addon=read_barcodes((FIXTURES/'ean_addon.png').read_bytes())
    assert payloads(addon)==[b'5901234123457']
    assert addon['backend']['ean_add_on_symbol']=='Ignore'
    gs1=read_barcodes((FIXTURES/'gs1.png').read_bytes())
    assert payloads(gs1)==[b'010950110153000310ABC123\x1d17260101']
    assert gs1['symbols'][0]['symbology_identifier']==']C1'


def test_control_bytes_and_cli_json_are_safe(tmp_path):
    content='AB\x00\x1b\n12'
    data=encoded(make_code('Code128',content))
    assert payloads(read_barcodes(data))==[content.encode()]
    path=tmp_path/'input.png';path.write_bytes(data)
    run=subprocess.run([sys.executable,'-m','opencv_bv_tools.barcode_reader',str(path),
                        '--format','Code128'],capture_output=True,timeout=5)
    assert run.returncode==0,run.stderr
    assert b'\x00' not in run.stdout and b'\x1b' not in run.stdout
    assert payloads(json.loads(run.stdout))==[content.encode()]
    assert path.read_bytes()==data
    bad=subprocess.run([sys.executable,'-m','opencv_bv_tools.barcode_reader',str(path),
                        '--timeout','nan'],capture_output=True,timeout=5)
    assert bad.returncode==2 and json.loads(bad.stderr)['error']


def test_multiple_identical_payloads_and_different_formats():
    canvas=Image.new('RGB',(1800,850),'white')
    tiles=[('Code128','ABC123',(20,20)),('Code128','ABC123',(900,20)),
           ('Code39','ABC123',(20,400))]
    for f,v,p in tiles:canvas.paste(make_code(f,v),p)
    result=read_barcodes(encoded(canvas),formats=['Code128','Code39'])
    assert payloads(result)==[b'ABC123']*3
    assert [s['format'] for s in result['symbols']]==['Code128','Code128','Code39']
    assert len({str(s['polygon']) for s in result['symbols']})==3
    limited=read_barcodes(encoded(canvas),formats=['Code128','Code39'],limits=BarcodeLimits(max_symbols=1))
    assert limited['search_status']=='resource_limited' and len(limited['symbols'])==1


@pytest.mark.parametrize('orientation',range(1,9))
def test_exif_polygon_transform(orientation):
    def exif_point(x,y,w,h,o):
        return {1:(x,y),2:(w-x,y),3:(w-x,h-y),4:(x,h-y),5:(y,x),
                6:(h-y,x),7:(h-y,w-x),8:(y,w-x)}[o]
    canvas=Image.new('RGB',(900,600),'white');canvas.paste(make_code('EAN8','0012345'),(30,80))
    baseline=read_barcodes(encoded(canvas))['symbols'][0]['polygon']
    exif=Image.Exif();exif[274]=orientation
    result=read_barcodes(encoded(canvas,exif=exif))
    assert payloads(result)==[b'00123457']
    assert result['image_size']==([900,600] if orientation<5 else [600,900])
    for x,y in baseline:
        a,b=exif_point(x,y,900,600,orientation)
        assert min((a-u)**2+(b-v)**2 for u,v in result['symbols'][0]['polygon'])<=16


@pytest.mark.parametrize('kind',['PNG','JPEG','WEBP'])
@pytest.mark.parametrize('mode',['1','L','P','RGB','RGBA','LA'])
def test_input_formats_modes(kind,mode):
    im=make_code('EAN8','0012345').convert(mode)
    if kind=='JPEG' and mode not in ('L','RGB'):im=im.convert('RGB')
    assert payloads(read_barcodes(encoded(im,kind)))==[b'00123457']


def test_alpha_and_icc_policy():
    im=make_code('EAN8','0012345').convert('RGBA');im.putalpha(0)
    assert payloads(read_barcodes(encoded(im)))==[]
    im.putalpha(255)
    result=read_barcodes(encoded(im,icc_profile=b'bad-profile'))
    assert payloads(result)==[b'00123457']
    assert any('ICC profile ignored' in w for w in result['warnings'])


@pytest.mark.parametrize('formats',[[],(),'', 'EAN13',['qr'],['EAN13','EAN13'],[None],[1],
                                   ['Code39','Code39Extended']])
def test_bad_filters(formats):
    with pytest.raises(BarcodeReadError):read_barcodes(b'x',formats=formats)


@pytest.mark.parametrize('kwargs',[
    {'limits':{}},{'cancelled':3},{'background':[255]*3},{'background':(True,1,1)},
    {'background':(256,0,0)}, {'limits':BarcodeLimits(max_pixels=True)},
    {'limits':BarcodeLimits(max_pixels=20_000_001)},{'limits':BarcodeLimits(max_dimension=0)},
    {'limits':BarcodeLimits(timeout_seconds=float('nan'))},
    {'limits':BarcodeLimits(timeout_seconds=float('inf'))},
    {'limits':BarcodeLimits(timeout_seconds=31)}, {'limits':BarcodeLimits(max_symbols=129)},
])
def test_bad_options(kwargs):
    with pytest.raises(BarcodeReadError):read_barcodes(b'x',**kwargs)


@pytest.mark.parametrize('data',[b'',bytearray(b'x'),'x'])
def test_invalid_input_type(data):
    with pytest.raises(BarcodeReadError):read_barcodes(data)


@pytest.mark.parametrize('kind',['PNG','WEBP'])
def test_animation_rejected(kind):
    im=make_code('Code128','A');other=make_code('Code128','B').resize(im.size)
    assert read_barcodes(encoded(im,kind,save_all=True,append_images=[other],duration=50))['search_status']=='failed'


@pytest.mark.parametrize('data',[b'not image',b'\x89PNG\r\n\x1a\n',encoded(Image.new('I',(10,10))),
    encoded(Image.new('RGB',(10,10)),'GIF'),encoded(Image.new('RGB',(10,10)),'TIFF')])
def test_invalid_images(data):
    assert read_barcodes(data)['stop_reason']=='invalid_or_unsupported_image'


def test_limits_and_output_caps():
    data=encoded(make_code('Code128','test limits'))
    for field in ('max_payload_bytes','max_total_payload_bytes','max_text_chars'):
        result=read_barcodes(data,limits=replace(BarcodeLimits(),**{field:2}))
        assert result['search_status']=='resource_limited' and not result['symbols']
    assert read_barcodes(data,limits=BarcodeLimits(max_pixels=10))['stop_reason']=='input_pixel_limit'
    assert read_barcodes(encoded(Image.new('L',(17000,1))))['stop_reason']=='input_dimension_limit'
    with pytest.raises(BarcodeReadError):read_barcodes(data,limits=BarcodeLimits(max_input_bytes=10))


def fake_worker(code,timeout=.25,cancelled=None):
    return _run_worker(b'x'*1_000_000,{'limits':asdict(BarcodeLimits()),'background':[255]*3,
        'formats':DEFAULT_FORMATS},timeout=timeout,cancelled=cancelled,command=[sys.executable,'-c',code])


@pytest.mark.parametrize('code,reason',[
    ('import time;time.sleep(10)','deadline_exceeded'),
    ("import sys,time;sys.stdout.write('{');sys.stdout.flush();time.sleep(10)",'deadline_exceeded'),
    ("import sys;sys.stdout.buffer.write(b'x'*2000000)",'worker_output_limit'),
    ("import sys;sys.stderr.buffer.write(b'x'*10000)",'worker_diagnostic_limit'),
    ('import os;os._exit(7)','worker_failed'),("print('{}')",'invalid_worker_response'),
])
def test_barcode_transport_faults_reap(code,reason):
    import psutil
    before={p.pid for p in psutil.Process().children()}
    assert fake_worker(code)['stop_reason']==reason
    assert {p.pid for p in psutil.Process().children()}==before


def test_cancellation_deadline_and_descriptor_stability():
    import psutil
    data=encoded(make_code('EAN13','590123412345'))
    assert read_barcodes(data,cancelled=lambda:True)['stop_reason']=='cancelled'
    assert read_barcodes(data,limits=BarcodeLimits(timeout_seconds=.00001))['stop_reason']=='deadline_exceeded'
    proc=psutil.Process();fds=proc.num_fds();children={p.pid for p in proc.children()}
    start=time.monotonic()
    assert fake_worker('import time;time.sleep(10)',cancelled=lambda:time.monotonic()-start>.08)['stop_reason']=='cancelled'
    for _ in range(6):assert payloads(read_barcodes(data))==[b'5901234123457']
    assert proc.num_fds()==fds and {p.pid for p in proc.children()}==children


def test_dependency_failure_and_worker_version(monkeypatch):
    from opencv_bv_tools.runtime import barcode_worker as worker
    data=encoded(make_code('Code128','dependency'))
    options={'limits':asdict(BarcodeLimits()),'background':[255]*3,'formats':DEFAULT_FORMATS}
    monkeypatch.setattr(worker,'version',lambda name:'0.0')
    assert worker.decode(data,options)['stop_reason']=='unsupported_decoder_version'
    monkeypatch.setitem(sys.modules,'zxingcpp',None)
    assert worker.decode(data,options)['stop_reason']=='missing_barcode_dependency'


def test_mutation_crops_stripes_and_text_do_not_invent_payloads():
    im=make_code('EAN13','590123412345')
    assert payloads(read_barcodes(encoded(im.crop((im.width//2,0,im.width,im.height)))))==[]
    blank=Image.new('RGB',(800,300),'white');d=ImageDraw.Draw(blank)
    d.text((20,20),'0123456789012 BARCODE TEST',fill='black')
    for x in range(20,780,8):d.rectangle((x,80,x+2,200),fill='black')
    assert payloads(read_barcodes(encoded(blank)))==[]
    data=encoded(im);rng=random.Random(20261103)
    for _ in range(12):
        bad=bytearray(data)
        for _ in range(5):bad[rng.randrange(len(bad))]=rng.randrange(256)
        result=read_barcodes(bytes(bad),limits=BarcodeLimits(timeout_seconds=.3))
        assert result['search_status'] in ('failed','completed','budget_exhausted')
        if result['symbols']:assert payloads(result)==[b'5901234123457']


def test_bwipp_fixture_hashes():
    for row in json.loads((FIXTURES/'manifest.json').read_text()):
        assert hashlib.sha256((FIXTURES/(row['id']+'.png')).read_bytes()).hexdigest()==row['sha256']


def test_mixed_polarities_preserve_separate_identical_symbols():
    tile=make_code('Code128','MIXED')
    canvas=Image.new('RGB',(tile.width+80,tile.height*2+100),'white')
    canvas.paste(tile,(20,20));canvas.paste(ImageOps.invert(tile),(20,tile.height+60))
    result=read_barcodes(encoded(canvas))
    assert payloads(result)==[b'MIXED',b'MIXED']
    assert [s['polarity'] for s in result['symbols']]==['visible','inverted']


@pytest.mark.parametrize('lengths',[None,[],[True],[3],[0],[82],[8,8],8,['8']])
def test_itf_rejects_missing_or_invalid_length_contract(lengths):
    with pytest.raises(BarcodeReadError):read_barcodes(b'x',formats=['ITF'],itf_lengths=lengths)


def test_itf_lengths_require_itf_selection():
    with pytest.raises(BarcodeReadError):read_barcodes(b'x',itf_lengths=[8])


def test_itf_length_constraint_filters_observed_truncations_not_checksums():
    im=make_code('ITF','00123456').rotate(15,expand=True,fillcolor='white')
    data=encoded(im)
    result=read_barcodes(data,formats=['ITF'],itf_lengths=[8])
    assert payloads(result)==[]
    assert len(result['undecoded'])==3
    assert all(x['status']=='rejected_length' for x in result['undecoded'])
    assert result['backend']['itf_lengths']==[8]
    permissive=read_barcodes(data,formats=['ITF'],itf_lengths=[2,4,6,8])
    assert set(payloads(permissive))=={b'001234',b'0012',b'00'}
    assert all(x['checksum_policy']=='optional_not_required' for x in permissive['symbols'])
    limited=read_barcodes(data,formats=['ITF'],itf_lengths=[8],limits=BarcodeLimits(max_symbols=1))
    assert limited['search_status']=='resource_limited' and len(limited['undecoded'])==1


def test_native_aligned_identical_region_merge_is_explicit():
    tile=make_code('Code128','SAME');canvas=Image.new('RGB',(900,tile.height*2+100),'white')
    canvas.paste(tile,(30,20));canvas.paste(tile,(30,tile.height+60))
    result=read_barcodes(encoded(canvas))
    assert payloads(result)==[b'SAME'] # Known backend merge; not a physical-symbol count.
    assert 'aligned_identical_barcodes_may_merge_into_one_region' in result['capability_limitations']
    polygon=result['symbols'][0]['polygon']
    assert max(p[1] for p in polygon)-min(p[1] for p in polygon)>tile.height


@pytest.mark.parametrize('format,value,expected',CASES)
def test_single_physical_code_not_returned_by_both_polarities(format,value,expected):
    # Targeted probe: no cross-polarity duplicates on these independent fixtures.
    result=read_barcodes(encoded(make_code(format,value)),formats=[format],
                         itf_lengths=[len(expected)] if format=='ITF' else None)
    assert payloads(result)==[expected]
    assert result['symbols'][0]['polarity']=='visible'


def test_worker_revalidates_itf_and_cli_requires_lengths(tmp_path):
    data=encoded(make_code('ITF','00123456'))
    options={'limits':asdict(BarcodeLimits()),'background':[255]*3,'formats':['ITF']}
    run=subprocess.run([sys.executable,'-m','opencv_bv_tools.runtime.barcode_worker'],
        input=json.dumps(options).encode()+b'\n'+data,capture_output=True,timeout=5)
    assert json.loads(run.stdout)['stop_reason']=='worker_error'
    path=tmp_path/'itf.png';path.write_bytes(data)
    command=[sys.executable,'-m','opencv_bv_tools.barcode_reader',str(path),'--format','ITF']
    bad=subprocess.run(command,capture_output=True,timeout=5)
    assert bad.returncode==2 and 'itf_lengths' in json.loads(bad.stderr)['error']
    good=subprocess.run(command+['--itf-length','8'],capture_output=True,timeout=5)
    assert good.returncode==0 and payloads(json.loads(good.stdout))==[b'00123456']
