"""Independent conversion regressions using synthesized in-memory fixtures."""
import io
import random
import struct
import unittest
import zlib

from PIL import Image, ImageCms, ImageOps, PngImagePlugin

from opencv_bv_tools.image_conversion import ConversionError
from opencv_bv_tools.image_conversion import convert_image


def encode(image, fmt, **kwargs):
    out = io.BytesIO()
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


def decoded(data):
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def png16_rgb():
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', 1, 1, 16, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(b'\x00\x12\x34\x56\x78\x9a\xbc')) + chunk(b'IEND', b'')


class IndependentConversions(unittest.TestCase):
    def test_palette_alpha_to_png_and_lossless_webp(self):
        image = Image.new('P', (3, 1))
        image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * 759)
        image.putdata([0, 1, 2])
        source = encode(image, 'PNG', transparency=bytes([0, 127, 255]))
        expected = list(decoded(source).convert('RGBA').get_flattened_data())
        for fmt in ('PNG', 'WEBP'):
            with self.subTest(format=fmt):
                out = decoded(convert_image(source, fmt).data).convert('RGBA')
                self.assertEqual(list(out.get_flattened_data()), expected)

    def test_rgb_trns_transparency_survives_lossless_outputs(self):
        image = Image.new('RGB', (2, 1))
        image.putdata([(10, 20, 30), (40, 50, 60)])
        source = encode(image, 'PNG', transparency=(10, 20, 30))
        expected = list(decoded(source).convert('RGBA').get_flattened_data())
        for fmt in ('PNG', 'WEBP'):
            with self.subTest(format=fmt):
                self.assertEqual(list(decoded(convert_image(source, fmt).data).convert('RGBA').get_flattened_data()), expected)

    def test_grayscale_normalizes_without_changing_values(self):
        image = Image.new('L', (3, 1)); image.putdata([0, 127, 255])
        result = convert_image(encode(image, 'PNG'), 'PNG')
        out = decoded(result.data)
        self.assertIn(out.mode, ('RGB', 'RGBA'))
        self.assertEqual(list(out.convert('RGB').get_flattened_data()), [(0, 0, 0), (127, 127, 127), (255, 255, 255)])

    def test_all_exif_orientations_applied_once_and_metadata_removed(self):
        image = Image.new('RGB', (3, 2))
        image.putdata([(240, 0, 0), (0, 240, 0), (0, 0, 240), (200, 200, 0), (0, 200, 200), (200, 0, 200)])
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                exif = image.getexif(); exif[274] = orientation
                source = encode(image, 'JPEG', quality=100, subsampling=0, exif=exif)
                expected = ImageOps.exif_transpose(decoded(source)).convert('RGB')
                result = convert_image(source, 'PNG')
                out = decoded(result.data).convert('RGB')
                self.assertEqual(out.size, expected.size)
                self.assertEqual(list(out.get_flattened_data()), list(expected.get_flattened_data()))
                self.assertEqual((result.width, result.height), expected.size)
                self.assertFalse(decoded(result.data).getexif())

    def test_rgb_16bit_png_rejected(self):
        with self.assertRaises(ConversionError):
            convert_image(png16_rgb(), 'PNG')

    def test_rgb_icc_preserved_but_text_and_exif_removed(self):
        image = Image.new('RGB', (2, 2), (80, 120, 160))
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        info = PngImagePlugin.PngInfo(); info.add_text('Comment', 'sensitive metadata')
        exif = image.getexif(); exif[270] = 'sensitive metadata'
        source = encode(image, 'PNG', icc_profile=profile, pnginfo=info, exif=exif)
        for fmt in ('PNG', 'WEBP', 'JPEG'):
            with self.subTest(format=fmt):
                result = convert_image(source, fmt)
                out = decoded(result.data)
                self.assertEqual(out.info.get('icc_profile'), profile)
                self.assertNotIn('Comment', out.info)
                self.assertFalse(out.getexif())
                self.assertNotIn('xmp', out.info)

    def test_invalid_rgb_icc_bytes_are_preserved_without_validation(self):
        # The API preserves opaque ICC metadata; it does not validate the profile.
        profile = b'not a valid ICC profile'
        source = encode(Image.new('RGB', (2, 2), 'red'), 'PNG', icc_profile=profile)
        for fmt in ('PNG', 'WEBP', 'JPEG'):
            with self.subTest(format=fmt):
                self.assertEqual(decoded(convert_image(source, fmt).data).info.get('icc_profile'), profile)

    def test_tagged_grayscale_is_rejected(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        source = encode(Image.new('L', (2, 2), 100), 'PNG', icc_profile=profile)
        with self.assertRaises(ConversionError):
            convert_image(source, 'PNG')

    def test_seeded_lossless_roundtrips(self):
        rng = random.Random(20261103)
        for mode in ('RGB', 'RGBA', 'L'):
            for index in range(4):
                image = Image.new(mode, (13, 9))
                channels = len(mode)
                values = [tuple(rng.randrange(256) for _ in range(channels)) for _ in range(13 * 9)]
                image.putdata([v[0] for v in values] if mode == 'L' else values)
                source = encode(image, 'PNG')
                for fmt in ('PNG', 'WEBP'):
                    with self.subTest(mode=mode, index=index, format=fmt):
                        out = decoded(convert_image(source, fmt).data).convert('RGBA')
                        self.assertEqual(list(out.get_flattened_data()), list(image.convert('RGBA').get_flattened_data()))

    def test_luminance_alpha_and_one_bit(self):
        for mode, pixels in [('LA', [(20, 0), (70, 90), (250, 255)]), ('1', [0, 255, 0])]:
            image = Image.new(mode, (3, 1)); image.putdata(pixels)
            source = encode(image, 'PNG')
            expected = list(decoded(source).convert('RGBA').get_flattened_data())
            for fmt in ('PNG', 'WEBP'):
                with self.subTest(mode=mode, format=fmt):
                    self.assertEqual(list(decoded(convert_image(source, fmt).data).convert('RGBA').get_flattened_data()), expected)

    def test_hidden_rgb_in_fully_transparent_pixels(self):
        image = Image.new('RGBA', (3, 1)); image.putdata([(10, 20, 30, 0), (200, 1, 99, 0), (5, 7, 9, 1)])
        for fmt in ('PNG', 'WEBP'):
            with self.subTest(format=fmt):
                self.assertEqual(list(decoded(convert_image(encode(image, 'PNG'), fmt).data).convert('RGBA').get_flattened_data()), list(image.get_flattened_data()))

    def test_opaque_rgba_converts_to_jpeg_without_background(self):
        source = encode(Image.new('RGBA', (8, 8), (150, 80, 40, 255)), 'PNG')
        out = decoded(convert_image(source, 'JPEG', quality=100).data).convert('RGB')
        self.assertTrue(all(abs(a-b) <= 3 for a,b in zip(out.getpixel((4,4)), (150,80,40))))

    def test_transparent_composite_jpeg_backgrounds(self):
        source = encode(Image.new('RGBA', (8, 8), (200, 40, 100, 128)), 'PNG')
        for background in ((0,0,0), (255,255,255), (30,140,220)):
            with self.subTest(background=background):
                expected = Image.alpha_composite(Image.new('RGBA', (8,8), background + (255,)), decoded(source)).convert('RGB').getpixel((0,0))
                out = decoded(convert_image(source, 'JPEG', quality=100, jpeg_background=background).data).convert('RGB')
                self.assertTrue(all(abs(a-b) <= 3 for a,b in zip(out.getpixel((4,4)), expected)))

    def test_transparent_palette_jpeg_requires_background(self):
        image = Image.new('P', (2, 1)); image.putpalette([255,0,0, 0,255,0] + [0]*762); image.putdata([0,1])
        source = encode(image, 'PNG', transparency=bytes([0,255]))
        with self.assertRaises(ConversionError):
            convert_image(source, 'JPEG')

    def test_lossy_webp_gradient_error_bound(self):
        image = Image.new('RGB', (96,64))
        image.putdata([((x*255)//95, (y*255)//63, ((x+y)*255)//158) for y in range(64) for x in range(96)])
        result = convert_image(encode(image, 'PNG'), 'WEBP', quality=90, webp_lossless=False)
        output = decoded(result.data).convert('RGB')
        diffs = [abs(a-b) for left,right in zip(image.get_flattened_data(), output.get_flattened_data()) for a,b in zip(left,right)]
        self.assertLessEqual(sum(diffs)/len(diffs), 5)
        self.assertLessEqual(max(diffs), 40)

    def test_alpha_jpeg_requires_background_and_composites(self):
        image = Image.new('RGBA', (8, 8), (255, 0, 0, 0))
        source = encode(image, 'PNG')
        with self.assertRaises(ConversionError):
            convert_image(source, 'JPEG')
        out = decoded(convert_image(source, 'JPEG', jpeg_background=(0, 200, 0), quality=100).data).convert('RGB')
        actual = out.getpixel((4, 4))
        for value, expected in zip(actual, (0, 200, 0)):
            self.assertLessEqual(abs(value - expected), 3)


if __name__ == '__main__':
    unittest.main()
