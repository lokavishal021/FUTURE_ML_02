
import os

__all__ = ["what"]

def what(file, h=None):
    f = None
    try:
        if h is None:
            if isinstance(file, (str, os.PathLike)):
                f = open(file, 'rb')
                h = f.read(32)
            else:
                location = file.tell()
                h = file.read(32)
                file.seek(location)
    except:
        return None
    finally:
        if f: f.close()

    if not h:
        return None

    for tf in tests:
        res = tf(h, f)
        if res:
            return res
    return None

def test_jpeg(h, f):
    """JPEG data in JFIF or Exif format"""
    if h[6:10] in (b'JFIF', b'Exif'):
        return 'jpeg'
    if h[:2] == b'\xff\xd8':
        return 'jpeg'

def test_png(h, f):
    if h[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'

def test_gif(h, f):
    """GIF ('87 and '89 variants)"""
    if h[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'

def test_tiff(h, f):
    """TIFF (can be in Motorola or Intel byte order)"""
    if h[:2] == b'MM' and h[2:4] == b'\x00*':
        return 'tiff'
    if h[:2] == b'II' and h[2:4] == b'*\x00':
        return 'tiff'

def test_rgb(h, f):
    """SGI ImgLib Files"""
    if h[:2] == b'\x01\xda':
        return 'rgb'

def test_pbm(h, f):
    """PBM (portable bitmap)"""
    if len(h) >= 3 and \
        h[0] == ord(b'P') and h[1] in b'14' and h[2] in b' \t\n\r':
        return 'pbm'

def test_pgm(h, f):
    """PGM (portable graymap)"""
    if len(h) >= 3 and \
        h[0] == ord(b'P') and h[1] in b'25' and h[2] in b' \t\n\r':
        return 'pgm'

def test_ppm(h, f):
    """PPM (portable pixmap)"""
    if len(h) >= 3 and \
        h[0] == ord(b'P') and h[1] in b'36' and h[2] in b' \t\n\r':
        return 'ppm'

def test_rast(h, f):
    """Sun raster file"""
    if h[:4] == b'\x59\xa6\x6a\x95':
        return 'rast'

def test_xbm(h, f):
    """X bitmap (X10 or X11)"""
    s = b'#define '
    if h[:len(s)] == s:
        return 'xbm'

def test_bmp(h, f):
    if h[:2] == b'BM':
        return 'bmp'

def test_webp(h, f):
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'

def test_exr(h, f):
    if h[:4] == b'\x76\x2f\x31\x01':
        return 'exr'

tests = [
    test_jpeg, test_png, test_gif, test_tiff, test_rgb,
    test_pbm, test_pgm, test_ppm, test_rast, test_xbm,
    test_bmp, test_webp, test_exr
]
