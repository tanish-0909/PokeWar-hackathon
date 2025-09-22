r"""
augment_composites.py

Run this *after* you have generated composites (the script you showed earlier).
It reads PNG/JPEG images from the composites folder and writes augmented
copies to an output folder (defaults to a subfolder named "augmented").

Features (configurable at top):
- Apply salt-and-pepper noise to ~25% of images (configurable probability)
- Overlay semi-transparent colored circles (30% opacity) in random places
- Apply random Gaussian blur
- Draw thin random squiggly lines across the image
- Randomly change brightness and saturation

The script does NOT change bounding boxes/annotations (these remain valid
because we only apply color/blur/noise overlays, not geometric transforms).

Usage:
    python augment_composites.py --input-dir "D:\Pokemon Project AI guild\composites" --output-dir "D:\Pokemon Project AI guild\augmented" --n-aug 2

"""

import os
import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm

# ---------------- CONFIG ----------------
# Defaults; you can override via CLI
DEFAULT_INPUT_DIR = r"D:\Pokemon Project AI guild\composites"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_INPUT_DIR, "augmented")
N_AUG_PER_IMAGE = 1  # how many augmented variants to make per input image
SEED = None

# Probabilities (per-variant) for each augmentation to be applied
SP_PROB = 0.25          # 25% of variants get salt & pepper noise
SP_AMOUNT = 0.05        # fraction of pixels affected when SP applied (5%)
CIRCLES_PROB = 0.8      # probability to add colored circles overlay
CIRCLES_COUNT = (16, 32)  # range (min,max) number of circles per overlay
CIRCLES_OPACITY = 0.20  # 30% opacity
CIRCLE_COLORS = [(255, 215, 0), (0, 200, 0), (255, 120, 0), (160, 32, 240)]  # yellow, green, orange, purple

BLUR_PROB = 0.1
BLUR_RADIUS_RANGE = (0.3, 1.0)

SQUIGGLE_PROB = 0.5
SQUIGGLE_COUNT = (1, 4)   # number of squiggly lines
SQUIGGLE_WIDTH = (1, 3)
SQUIGGLE_OPACITY = 0.5    # line opacity (0-1)

BRIGHT_SAT_PROB = 0.9
BRIGHTNESS_RANGE = (0.7, 1.3)  # multiply factor
SATURATION_RANGE = (0.7, 1.3)

# Filetypes to process
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
# -----------------------------------------


def add_salt_and_pepper(img: Image.Image, amount: float = 0.05) -> Image.Image:
    """Add salt and pepper noise to an RGB image. amount is fraction of pixels to change."""
    if img.mode != "RGB":
        base = img.convert("RGB")
    else:
        base = img.copy()
    arr = np.array(base)
    h, w, c = arr.shape
    num_pixels = int(h * w * amount)
    # Salt (white)
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    arr[ys, xs] = 255
    # Pepper (black) for another set
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    arr[ys, xs] = 0
    return Image.fromarray(arr)


def add_colored_circles(img: Image.Image, count_range=(2, 5), opacity=0.3, colors=None) -> Image.Image:
    """Overlay semi-transparent colored circles at random positions/sizes."""
    if colors is None:
        colors = CIRCLE_COLORS
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    n = random.randint(*count_range)
    for _ in range(n):
        radius = random.randint(int(min(w, h) * 0.05), int(min(w, h) * 0.10))
        cx = random.randint(-radius // 2, w + radius // 2)
        cy = random.randint(-radius // 2, h + radius // 2)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        color = random.choice(colors)
        alpha = int(round(255 * opacity))
        draw.ellipse(bbox, fill=(color[0], color[1], color[2], alpha))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert(img.mode)


def add_gaussian_blur(img: Image.Image, radius_range=(0.5, 2.0)) -> Image.Image:
    r = random.uniform(*radius_range)
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def add_squiggly_lines(img: Image.Image, count_range=(1, 3), width_range=(1, 2), opacity=0.6) -> Image.Image:
    """Draw thin squiggly lines across the image. Lines are semi-transparent."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    n = random.randint(*count_range)
    for _ in range(n):
        # generate a random polyline with controlled curvature
        pts = []
        num_pts = random.randint(3, 6)
        for i in range(num_pts):
            x = int(w * (i / (num_pts - 1)))
            # add vertical jitter
            y = random.randint(int(h * 0.0), int(h * 1.0))
            pts.append((x, y))
        line_color = tuple(random.randint(0, 255) for _ in range(3))
        alpha = int(round(255 * opacity))
        line_width = random.randint(*width_range)
        # draw a line between pts
        draw.line(pts, fill=(line_color[0], line_color[1], line_color[2], alpha), width=line_width)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert(img.mode)


def adjust_brightness_and_saturation(img: Image.Image, brightness_range=(0.7, 1.3), saturation_range=(0.7, 1.3)) -> Image.Image:
    # Brightness
    b_factor = random.uniform(*brightness_range)
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    # Saturation (Color in PIL)
    s_factor = random.uniform(*saturation_range)
    img = ImageEnhance.Color(img).enhance(s_factor)
    return img


def process_image_variant(img_path: Path, out_path: Path, variant_idx: int, cfg: dict):
    img = Image.open(img_path).convert("RGB")
    # apply augmentations based on probabilities

    # copy to avoid modifying original
    aug = img

    if random.random() < cfg["circles_prob"]:
        aug = add_colored_circles(aug, count_range=cfg["circles_count"], opacity=cfg["circles_opacity"], colors=cfg["colors"])

    if random.random() < cfg["blur_prob"]:
        aug = add_gaussian_blur(aug, radius_range=cfg["blur_radius_range"])

    if random.random() < cfg["squiggle_prob"]:
        aug = add_squiggly_lines(aug, count_range=cfg["squiggle_count"], width_range=cfg["squiggle_width"], opacity=cfg["squiggle_opacity"])

    if random.random() < cfg["sp_prob"]:
        aug = add_salt_and_pepper(aug, amount=cfg["sp_amount"])

    if random.random() < cfg["bright_sat_prob"]:
        aug = adjust_brightness_and_saturation(aug, brightness_range=cfg["brightness_range"], saturation_range=cfg["saturation_range"])

    # save with suffix
    out_name = f"{img_path.stem}_aug{variant_idx}{img_path.suffix}"
    out_file = out_path / out_name
    aug.save(out_file, quality=95)
    return out_file


def main_cli():
    parser = argparse.ArgumentParser(description="Augment composite images for robustness")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-aug", type=int, default=N_AUG_PER_IMAGE, help="number of augmented variants per image")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed (optional)")
    parser.add_argument("--sp-prob", type=float, default=SP_PROB, help="probability to apply salt&pepper to a variant")
    parser.add_argument("--sp-amount", type=float, default=SP_AMOUNT, help="salt&pepper amount (fraction of pixels)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    cfg = {
        "sp_prob": args.sp_prob,
        "sp_amount": args.sp_amount,
        "circles_prob": CIRCLES_PROB,
        "circles_count": CIRCLES_COUNT,
        "circles_opacity": CIRCLES_OPACITY,
        "colors": CIRCLE_COLORS,
        "blur_prob": BLUR_PROB,
        "blur_radius_range": BLUR_RADIUS_RANGE,
        "squiggle_prob": SQUIGGLE_PROB,
        "squiggle_count": SQUIGGLE_COUNT,
        "squiggle_width": SQUIGGLE_WIDTH,
        "squiggle_opacity": SQUIGGLE_OPACITY,
        "bright_sat_prob": BRIGHT_SAT_PROB,
        "brightness_range": BRIGHTNESS_RANGE,
        "saturation_range": SATURATION_RANGE,
    }

    # gather images
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    if not files:
        print("No images found in", input_dir)
        return

    print(f"Found {len(files)} images. Writing augmented images to {output_dir} (n_aug={args.n_aug})")

    for img_path in tqdm(files, desc="Augmenting images"):
        for i in range(args.n_aug):
            process_image_variant(img_path, output_dir, i + 1, cfg)

    print("Augmentation complete.")


if __name__ == "__main__":
    main_cli()
