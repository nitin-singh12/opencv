# Independent manual review

Reviewed the three source images before considering detector output. Each contains one visible face. Boxes and uncertainty are in manual-annotations.json; they cover the visible facial region, including ears where visible, while largely excluding hair and headwear. Boundaries are approximate and should not be treated as pixel-accurate segmentation. Camera image is a side profile; the glasses and hat in grace_hopper.jpg cover parts of the face.

Source context supplied with task: scikit-image v0.25.2 astronaut (NASA public domain) and camera (CC0, Lav Varshney); matplotlib v3.10.7 grace_hopper. The JPEG comment identifies http://commons.wikimedia.org/wiki/File:Grace_Hopper.jpg as its source; the image license and provenance were not independently verified.
