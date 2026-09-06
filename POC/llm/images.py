"""A figure on disk, as something a chat model will accept.

Kept apart from the chains because it is the one piece of the vision call that
is not a model concern at all: figures are rendered at 200 dpi, which is far
more than a vision model reads and is billed by the tile.
"""

import base64

import pymupdf

from .constants import IMAGE_MIME_TYPE, JPEG_QUALITY, MAX_IMAGE_EDGE


def encode_image(image, max_edge=MAX_IMAGE_EDGE, quality=JPEG_QUALITY):
    """Downscale an image to a JPEG data URI.

    Halving until the long edge fits `max_edge` keeps axis labels legible while
    cutting the image token cost several-fold.
    """
    pixmap = pymupdf.Pixmap(image)        # accepts a path or the encoded bytes

    while max(pixmap.width, pixmap.height) > max_edge:
        pixmap.shrink(1)                      # halves both edges

    if pixmap.colorspace is None or pixmap.colorspace.n != 3:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)   # CMYK/greyscale -> RGB
    if pixmap.alpha:
        # A fifth of the rendered figures carry an alpha channel, and converting
        # colorspace does not drop it. JPEG has no alpha, so flatten explicitly.
        pixmap = pymupdf.Pixmap(pixmap, 0)

    payload = base64.b64encode(pixmap.tobytes("jpeg", jpg_quality=quality)).decode()
    return f"data:{IMAGE_MIME_TYPE};base64,{payload}"
