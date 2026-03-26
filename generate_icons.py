"""
Generates Android app icons for Jio Antenna Chatbot.
Run from repo root: python3 generate_icons.py
"""
from PIL import Image, ImageDraw
import os

sizes = {
    "mipmap-mdpi":    48,
    "mipmap-hdpi":    72,
    "mipmap-xhdpi":   96,
    "mipmap-xxhdpi":  144,
    "mipmap-xxxhdpi": 192,
}

base = "android/app/src/main/res"

def draw_chatbot_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    p = size // 12

    # Background — Jio blue
    draw.rounded_rectangle([0, 0, size, size], radius=size // 5,
                            fill=(26, 115, 232))

    # White chat bubble
    bx1 = p * 2
    by1 = int(size * 0.18)
    bx2 = size - p * 2
    by2 = int(size * 0.65)
    r_bub = size // 8
    draw.rounded_rectangle([bx1, by1, bx2, by2], radius=r_bub, fill=(255, 255, 255))

    # Bubble tail (triangle bottom-left)
    tx = bx1 + size // 8
    draw.polygon([
        (tx, by2),
        (tx - size // 9, by2 + size // 7),
        (tx + size // 8, by2)
    ], fill=(255, 255, 255))

    # Three dots inside bubble
    dot_r = max(2, size // 18)
    cy_dot = (by1 + by2) // 2
    spacing = (bx2 - bx1) // 4
    for i in range(3):
        cx_dot = bx1 + spacing * (i + 1)
        draw.ellipse([cx_dot - dot_r, cy_dot - dot_r,
                      cx_dot + dot_r, cy_dot + dot_r],
                     fill=(26, 115, 232))

    # Red accent bar at bottom
    bar_h = max(4, size // 10)
    draw.rounded_rectangle([p * 2, size - p - bar_h, size - p * 2, size - p],
                            radius=bar_h // 2, fill=(213, 0, 0))
    return img


for folder, size in sizes.items():
    os.makedirs(f"{base}/{folder}", exist_ok=True)

    img = draw_chatbot_icon(size)

    # Square icon
    sq = Image.new("RGB", (size, size), (26, 115, 232))
    sq.paste(img, mask=img.split()[3])
    sq.save(f"{base}/{folder}/ic_launcher.png")

    # Round icon — circle crop
    rnd = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size, size], fill=255)
    rnd.paste(img, mask=mask)
    rnd.save(f"{base}/{folder}/ic_launcher_round.png")

    print(f"Generated {folder}: {size}x{size}px")

print("All chatbot icons generated successfully")
