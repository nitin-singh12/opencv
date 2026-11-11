# Public face fixtures and independent annotations

The downloaded byte hashes and source URLs are in `provenance.json`.

- `astronaut.png`: scikit-image v0.25.2, NASA photograph. The
  [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut)
  describes it as public domain. The original has an ICC profile and is rejected
  by the runtime. The pilot explicitly converts that profile to sRGB before
  generating its untagged derived inputs. This is fixture preparation, not a
  runtime profile-removal fallback.
- `camera.png`: scikit-image v0.25.2, CC0 photograph by Lav Varshney, documented
  in [scikit-image](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.camera).
- `grace_hopper.jpg`: Matplotlib v3.10.7 derivative of the
  [US Navy official photograph by James S. Davis](https://commons.wikimedia.org/wiki/File:Grace_Hopper.jpg),
  described there as public domain. The fixture's JPEG comment cites that source.

Terra independently examined all three source images and annotated visible face
rectangles before receiving detector results. `manual-annotations.json` includes
source hashes, dimensions, uncertainty and occlusion notes. The boxes are
approximate visible-feature rectangles, not segmentation masks. Derived variants
reuse these three sources and do not establish demographic coverage, general
detector recall, anonymization strength or a held-out evaluation.
