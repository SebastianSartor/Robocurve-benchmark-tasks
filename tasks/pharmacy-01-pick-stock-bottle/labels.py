"""Generate bottle labels and the printed order card at known physical type sizes."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from layout import CARD_SIZE, SLOT_SIZE, ZONE_SIZE


NAVY = "#182b46"
PAPER = "#fffdf7"
TEAL = "#1d6376"
PX_PER_MM = 16


def pixels(mm):
    return round(mm * PX_PER_MM)


def canvas(width_mm, height_mm, color):
    image = Image.new("RGB", (pixels(width_mm), pixels(height_mm)), color)
    return image, ImageDraw.Draw(image)


def typeface(points):
    size = pixels(points * 25.4 / 72)
    for candidate in ("DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise RuntimeError("Install DejaVu Sans Bold or Arial Bold to generate the labels")


def text(draw, image, value, y, points, color=NAVY):
    draw.text((image.width / 2, pixels(y)), value, fill=color, font=typeface(points), anchor="mm")


def bottle_label(stock, *, cap):
    width, height = (39, 20) if cap else (51, 34)
    image, draw = canvas(width, height, PAPER)
    if cap:
        draw.rectangle((0, 0, image.width, pixels(2)), fill=TEAL)
        text(draw, image, stock.name, 7, 15)
        text(draw, image, stock.strength, 14, 16)
    else:
        draw.rectangle((0, 0, image.width, pixels(7)), fill=TEAL)
        text(draw, image, "PHARMACY STOCK", 3.5, 14, "white")
        text(draw, image, stock.name, 14, 18)
        text(draw, image, stock.strength, 22, 19)
        text(draw, image, "SIMULATION", 30, 14)
    return image


def prescription(stock):
    image, draw = canvas(CARD_SIZE[1] * 1000, CARD_SIZE[0] * 1000, PAPER)
    draw.rectangle((0, 0, image.width, pixels(21)), fill=NAVY)
    text(draw, image, "PRESCRIPTION", 10, 37, "white")
    text(draw, image, stock.name, 45, 55)
    text(draw, image, stock.strength, 65, 49)
    text(draw, image, "SIMULATION ONLY", 86, 25, TEAL)
    return image


def marker(width_mm, height_mm, background, outline, legend=None):
    image, draw = canvas(width_mm, height_mm, background)
    inset = pixels(0.8)
    draw.rectangle((inset, inset, image.width - inset, image.height - inset), outline=outline, width=pixels(1.7))
    if legend:
        text(draw, image, legend, 11, 39, outline)
    return image


def prepare(layout, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for stock in layout.bottles:
        files[stock.key] = {}
        for cap in (False, True):
            side = "cap" if cap else "front"
            path = directory / f"{stock.key}-{side}.png"
            bottle_label(stock, cap=cap).save(path)
            files[stock.key][side] = path
    for name, image in (
        ("order", prescription(layout.target)),
        ("staging", marker(*(side * 1000 for side in ZONE_SIZE[::-1]), "#267347", "#e9f5d5", "STAGING")),
        ("slot", marker(*(side * 1000 for side in SLOT_SIZE[::-1]), "#8292a4", "#fcdaa0")),
    ):
        path = directory / f"{name}.png"
        image.save(path)
        files[name] = path
    return files
