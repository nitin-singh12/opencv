"""Independent test/pilot encoders, never imported by the runtime package."""
from io import BytesIO
from pathlib import Path
import barcode
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw

FIXTURES = Path(__file__).resolve().parents[1] / 'tests/fixtures/barcode'
CASES = [
    ('EAN13','590123412345',b'5901234123457'),
    ('EAN8','0012345',b'00123457'),
    ('UPCA','01234567890',b'0012345678905'),
    ('UPCE','0123456',b'0012345000065'),
    ('Code128','00AbC-123',b'00AbC-123'),
    ('Code39','ABC123',b'ABC123'),
    ('Code39Extended','abc',b'abc'),
    ('ITF','00123456',b'00123456'),
]


def encoded(image, kind='PNG', **kwargs):
    stream = BytesIO()
    image.save(stream, format=kind, **kwargs)
    return stream.getvalue()


def make_code(format, value, **kwargs):
    if format in ('UPCE','Code39Extended'):
        path = FIXTURES / (('upce_'+value if format == 'UPCE' else 'code39ext')+'.png')
        with Image.open(path) as im:
            return im.convert('RGB')
    name = {'EAN13':'ean13','EAN8':'ean8','UPCA':'upca','Code128':'code128',
            'Code39':'code39','ITF':'itf'}[format]
    if format == 'Code39': kwargs.setdefault('add_checksum',False)
    obj = barcode.get_barcode_class(name)(value,writer=ImageWriter(),**kwargs)
    return obj.render({'module_width':.4,'module_height':15,'quiet_zone':5,'write_text':False}).convert('RGB')


def bits_image(bits, scale=4):
    # Draw independently encoded module bits; no decoder used for expected labels.
    image = Image.new('RGB',((len(bits)+24)*scale,180),'white')
    draw=ImageDraw.Draw(image)
    for i,bit in enumerate(bits):
        if bit == '1': draw.rectangle(((i+12)*scale,20,(i+13)*scale-1,159),fill='black')
    return image


def invalid_checksum(format):
    if format in ('EAN13','EAN8','UPCA'):
        kind = 'ean8' if format == 'EAN8' else 'ean13'
        valid = {'EAN13':'5901234123457','EAN8':'00123457','UPCA':'0012345678905'}[format]
        wrong=valid[:-1]+str((int(valid[-1])+1)%10)
        obj=barcode.get_barcode_class(kind)(wrong,no_checksum=True)
        assert obj.get_fullcode()==wrong
        return bits_image(obj.build()[0])
    from barcode.charsets import code128
    obj=barcode.get_barcode_class('code128')('CHECK123')
    codes=obj._build()
    codes.append((obj._calculate_checksum(codes)+1)%103)
    return bits_image(''.join(code128.CODES[n] for n in codes)+code128.STOP+'11')
