#!/usr/bin/env python3
"""Draw the game's icon - a briefcase with a clock on it, running late for work.

    python packaging/make_icon.py                 # writes src/gettowork/assets/icon.png
    python packaging/make_icon.py --out icon.png  # somewhere else

The icon is pixel art: a 32 x 32 grid of "big pixels", each drawn as an
8 x 8 block, so the finished picture is 256 x 256. It uses only the standard
library - including a tiny PNG writer (:func:`png_bytes`) - so anyone can
re-draw it without installing anything.

The PNG it writes is committed to the repository. The game window uses it as
its icon, and the build recipe (``packaging/gettowork.spec``) turns it into a
Windows ``.ico`` / macOS ``.icns`` with Pillow when it builds the game.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path
from typing import Callable, Optional, Sequence

GRID = 32  # big pixels per side
SCALE = 8  # real pixels per big pixel
SIZE = GRID * SCALE  # 256
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src" / "gettowork" / "assets" / "icon.png"

Color = tuple[int, int, int, int]  # red, green, blue, alpha (0-255)
CLEAR: Color = (0, 0, 0, 0)


def rgb(hex_color: str) -> Color:
    """'#rrggbb' -> an opaque RGBA colour."""
    value = hex_color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)


# The palette: the game's dark window, its magenta/lilac accents, a leather briefcase.
OUTLINE = rgb("#1b1226")
SKY_TOP = rgb("#43336b")
SKY_BOTTOM = rgb("#261d3d")
SHADOW = rgb("#1c1530")
SPEED = rgb("#d7a8f0")  # the window's accent colour
PUFF = rgb("#7c6aa6")
LEATHER = rgb("#c9772f")
LEATHER_LIGHT = rgb("#eb9d52")
LEATHER_DARK = rgb("#94521c")
HANDLE = rgb("#5e3413")
BRASS = rgb("#f5c451")
FACE = rgb("#f7f3e8")
HANDS = rgb("#2b1b10")
SNEAKER = rgb("#e0457b")  # magenta running shoes
LEGS = rgb("#f0d9b5")  # light enough to see against the dark sky


class Canvas:
    """A GRID x GRID picture of big pixels (row by row, top to bottom)."""

    def __init__(self) -> None:
        self.pixels: list[list[Color]] = [[CLEAR] * GRID for _ in range(GRID)]

    def put(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < GRID and 0 <= y < GRID:
            self.pixels[y][x] = color

    def fill(self, inside: Callable[[int, int], bool], color: Color) -> None:
        """Paint every big pixel for which ``inside(x, y)`` is true."""
        for y in range(GRID):
            for x in range(GRID):
                if inside(x, y):
                    self.put(x, y, color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        """A filled rectangle; both corners are included."""
        self.fill(lambda x, y: x0 <= x <= x1 and y0 <= y <= y1, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        """A one-pixel line (simple DDA: good enough for short pixel-art strokes)."""
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(steps + 1):
            self.put(round(x0 + (x1 - x0) * i / steps), round(y0 + (y1 - y0) * i / steps), color)


def rounded(x0: int, y0: int, x1: int, y1: int, radius: float) -> Callable[[int, int], bool]:
    """Is a big pixel inside this rectangle with rounded corners?"""

    def inside(x: int, y: int) -> bool:
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        # Distance from the pixel's centre to the nearest corner circle's centre.
        cx = min(max(x + 0.5, x0 + radius), x1 + 1 - radius)
        cy = min(max(y + 0.5, y0 + radius), y1 + 1 - radius)
        return (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= radius**2

    return inside


def disc(cx: float, cy: float, radius: float) -> Callable[[int, int], bool]:
    return lambda x, y: (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= radius**2


def mix(a: Color, b: Color, amount: float) -> Color:
    """Blend colour a towards b (0 = a, 1 = b)."""
    return tuple(round(p + (q - p) * amount) for p, q in zip(a, b))  # type: ignore[return-value]


def draw() -> Canvas:
    """Paint the icon, back to front."""
    c = Canvas()

    # 1. A rounded square "sky" with a gentle top-to-bottom shading.
    frame = rounded(1, 1, 30, 30, 6)
    c.fill(frame, OUTLINE)
    body = rounded(2, 2, 29, 29, 5)
    for y in range(GRID):
        band = mix(SKY_TOP, SKY_BOTTOM, min(1.0, max(0.0, (y - 2) / 27)))
        c.fill(lambda x, yy, y=y: yy == y and body(x, yy), band)

    # 2. A shadow on the ground, and dust puffs kicked up behind the runner.
    c.fill(lambda x, y: y == 28 and 9 <= x <= 25, SHADOW)
    c.fill(disc(6.5, 26.5, 1.3), PUFF)
    c.fill(disc(4.5, 23.5, 1.0), PUFF)

    # 3. Speed lines: this briefcase is in a hurry.
    c.line(3, 12, 7, 12, SPEED)
    c.line(2, 16, 6, 16, SPEED)
    c.line(4, 20, 7, 20, SPEED)

    # 4. Running legs (one forward, one back) in magenta sneakers.
    c.line(13, 24, 11, 26, LEGS)
    c.line(14, 24, 12, 26, LEGS)
    c.rect(9, 26, 12, 27, SNEAKER)
    c.line(21, 24, 23, 25, LEGS)
    c.line(22, 24, 24, 25, LEGS)
    c.rect(23, 25, 26, 26, SNEAKER)

    # 5. The handle: a loop on top (the gap in the middle shows the sky).
    c.rect(14, 5, 21, 6, OUTLINE)
    c.rect(14, 5, 15, 9, OUTLINE)
    c.rect(20, 5, 21, 9, OUTLINE)
    c.rect(15, 6, 20, 6, HANDLE)
    c.rect(15, 6, 15, 9, HANDLE)
    c.rect(20, 6, 20, 9, HANDLE)

    # 6. The briefcase: outline, leather, a light top edge and a darker bottom.
    c.fill(rounded(8, 9, 27, 23, 2.5), OUTLINE)
    leather = rounded(9, 10, 26, 22, 1.8)
    c.fill(leather, LEATHER)
    c.fill(lambda x, y: y == 10 and leather(x, y), LEATHER_LIGHT)
    c.fill(lambda x, y: y >= 20 and leather(x, y), LEATHER_DARK)
    c.rect(11, 11, 11, 21, LEATHER_DARK)  # two straps
    c.rect(24, 11, 24, 21, LEATHER_DARK)
    c.rect(11, 10, 11, 11, BRASS)  # strap buckles
    c.rect(24, 10, 24, 11, BRASS)

    # 7. The clock on the front, reading "way too late". (Its centre is the
    #    middle of big pixel (17, 16), so the ticks sit evenly around it.)
    c.fill(disc(17.5, 16.5, 5.0), OUTLINE)
    c.fill(disc(17.5, 16.5, 4.1), FACE)
    for x, y in ((17, 13), (20, 16), (17, 19), (14, 16)):  # 12, 3, 6 and 9 o'clock ticks
        c.put(x, y, LEATHER_DARK)
    c.line(17, 16, 17, 14, HANDS)  # minute hand, pointing at 12
    c.line(17, 16, 19, 17, HANDS)  # hour hand, a little past 4
    c.put(17, 16, SNEAKER)  # centre pin

    # 8. A glint of light on the clock glass.
    c.put(15, 13, (255, 255, 255, 255))
    return c


def render(canvas: Optional[Canvas] = None, scale: int = SCALE) -> tuple[int, int, bytes]:
    """The picture as ``(width, height, RGBA bytes)``, each big pixel repeated ``scale`` times."""
    canvas = canvas or draw()
    rows = []
    for row in canvas.pixels:
        line = b"".join(bytes(color) * scale for color in row)
        rows.extend([line] * scale)
    return GRID * scale, GRID * scale, b"".join(rows)


def png_bytes(width: int, height: int, rgba: bytes) -> bytes:
    """Encode RGBA pixels as a PNG file.

    A PNG is a signature followed by "chunks" (length, 4-letter type, data,
    CRC checksum): IHDR (size and pixel format), IDAT (the zlib-compressed
    rows, each starting with a filter byte - 0 means "stored as is") and IEND.
    """
    if len(rgba) != width * height * 4:
        raise ValueError("rgba must hold width * height * 4 bytes")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    stride = width * 4
    raw = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8 bits per channel, RGBA, no interlace
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def read_png_rgba(data: bytes) -> tuple[int, int, bytes]:
    """Decode a PNG written by :func:`png_bytes` back to ``(width, height, RGBA bytes)``.

    Only understands what this script writes (8-bit RGBA, filter 0) - enough
    for the tests to check that the committed icon matches the drawing.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG file")
    pos, width, height, idat = 8, 0, 0, b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", body[:10])
            if (depth, color_type) != (8, 6):
                raise ValueError("only 8-bit RGBA PNGs are supported")
        elif kind == b"IDAT":
            idat += body
    raw = zlib.decompress(idat)
    stride = width * 4
    rows = []
    for y in range(height):
        start = y * (stride + 1)
        if raw[start] != 0:
            raise ValueError("only unfiltered rows are supported")
        rows.append(raw[start + 1:start + 1 + stride])
    return width, height, b"".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the Get To Work icon (256 x 256 PNG).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"where to write it (default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(png_bytes(*render()))
    print(f"Wrote {args.out} ({SIZE} x {SIZE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
