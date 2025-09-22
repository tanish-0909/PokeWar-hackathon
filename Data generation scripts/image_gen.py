import os
import csv
import json
import random
from PIL import Image, ImageOps

# --------- USER CONFIG ----------
pokemon_root = r"D:\Pokemon Project AI guild\Ref images"   # subfolders: pikachu, bulbasaur, charizard, mewtwo (case-insensitive)
background_dir = r"D:\Pokemon Project AI guild\cropped_bg_images"  # 640x480 backgrounds
save_dir = r"D:\Pokemon Project AI guild\composites"
os.makedirs(save_dir, exist_ok=True)

# output paths
csv_path = os.path.join(save_dir, "annotations.csv")
coco_json_path = os.path.join(save_dir, "coco_annotations.json")

# Class mapping (use exactly these IDs)
CLASS_NAME_TO_ID = {
    "pikachu": 1,
    "charizard": 2,
    "bulbasaur": 3,
    "mewtwo": 4,
}
# canonical names by id for COCO / CSV
ID_TO_NAME = {v: k.capitalize() for k, v in CLASS_NAME_TO_ID.items()}

# Behaviour / tuning
BG_SIZE = (640, 480)            # expected background size (width, height)
GAP_PX = 45                      # minimum gap between sprites (in pixels)
MIN_SPRITE_W = 45               # min sprite width in px
MAX_SPRITE_W = 500              # max sprite width in px
MAX_PLACEMENT_TRIES = 200       # tries to place each sprite before giving up

# Rotation: sprites will be rotated by a random angle in [-ROTATION_MAX_DEG, ROTATION_MAX_DEG]
ROTATION_MAX_DEG = 180

# Flip probability (horizontal flip)
FLIP_PROB = 0.5

# Probability distribution for counts per class (0..5) - skewed to 0 or 1
COUNT_WEIGHTS = [0.40, 0.35, 0.10, 0.07, 0.05, 0.03]
# ---------------------------------

def load_pokemon_library(root, allowed_lower_names):
    """
    Return dict: {lower_class_name: [list of absolute sprite paths]}
    Only includes classes present in CLASS_NAME_TO_ID (case-insensitive).
    """
    classes = {}
    if not os.path.isdir(root):
        print("Pokemon root not found:", root)
        return classes

    # list folders in root and map lowercase -> actual folder name
    folder_map = {}
    for d in os.listdir(root):
        full = os.path.join(root, d)
        if os.path.isdir(full):
            folder_map[d.lower()] = d

    for lname in allowed_lower_names:
        if lname in folder_map:
            actual = folder_map[lname]
            fullpath = os.path.join(root, actual)
            sprites = [os.path.join(fullpath, f) for f in os.listdir(fullpath)
                       if f.lower().endswith((".png", ".webp", ".tiff", ".bmp"))]
            if sprites:
                classes[lname] = sprites
            else:
                print(f"Warning: no sprite images found in folder {fullpath}")
        else:
            print(f"Warning: expected folder for '{lname}' not found in {root}")
    return classes

def sample_count():
    """Sample 0..5 using weights above"""
    return random.choices(range(6), weights=COUNT_WEIGHTS, k=1)[0]

def rects_intersect(r1, r2):
    """r: (xmin,ymin,xmax,ymax). Return True if overlaps (edge-touch counts as overlap)."""
    ax1, ay1, ax2, ay2 = r1
    bx1, by1, bx2, by2 = r2
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def inflate_rect(rect, pad):
    xmin, ymin, xmax, ymax = rect
    return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)

def place_sprite(bg_w, bg_h, sprite_w, sprite_h, existing_boxes, gap, tries=MAX_PLACEMENT_TRIES):
    """Try to find a top-left (x,y) where [x, y, x+w, y+h] doesn't overlap existing boxes inflated by gap."""
    for _ in range(tries):
        x = random.randint(0, max(0, bg_w - sprite_w))
        y = random.randint(0, max(0, bg_h - sprite_h))
        candidate = (x, y, x + sprite_w, y + sprite_h)
        candidate_inflated = inflate_rect(candidate, gap)
        collision = False
        for box in existing_boxes:
            if rects_intersect(candidate_inflated, inflate_rect(box, 0)):
                collision = True
                break
        if not collision:
            return candidate
    return None

def resize_sprite_keep_aspect(sprite: Image.Image, target_w: int):
    """Resize sprite so width == target_w and aspect ratio preserved."""
    w, h = sprite.size
    if w == 0:  # guard
        return sprite
    scale = target_w / w
    new_h = max(1, int(round(h * scale)))
    return sprite.resize((target_w, new_h), Image.Resampling.LANCZOS)

def apply_random_rotation_and_crop(sprite: Image.Image, max_deg: float):
    """
    Rotate sprite by a random angle in [-max_deg, max_deg], with expand=True,
    then crop transparent borders using alpha channel. Returns cropped RGBA sprite.
    """
    angle = random.uniform(-max_deg, max_deg)
    rotated = sprite.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    if rotated.mode in ("RGBA", "LA"):
        alpha = rotated.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            cropped = rotated.crop(bbox)
            return cropped
        else:
            return None
    else:
        bbox = rotated.getbbox()
        if bbox:
            return rotated.crop(bbox)
        return None

def compose_for_background(bg_path, classes_dict, out_path, image_id):
    """
    Compose one composite image and return:
      out_file (path) and annotations list: [ (class_id, class_name, xmin, ymin, xmax, ymax), ... ]
    """
    try:
        bg = Image.open(bg_path).convert("RGBA")
    except Exception as e:
        print(f"Skipping background {bg_path}: can't open ({e})")
        return None, []

    # ensure bg size
    if bg.size != BG_SIZE:
        bg = bg.resize(BG_SIZE, Image.Resampling.LANCZOS)
    bg_w, bg_h = bg.size

    # pick counts per class (using lowercase keys from CLASS_NAME_TO_ID)
    counts = {lname: sample_count() for lname in classes_dict.keys()}

    # it's okay to have 0 or 1 pokemons overall per your last note: we keep current distribution

    placed_boxes = []        # store actual placed boxes (xmin,ymin,xmax,ymax)
    annotations = []         # store tuples (class_id, class_name, xmin,ymin,xmax,ymax)

    for lname, ccount in counts.items():
        if ccount <= 0:
            continue
        sprites = classes_dict[lname]
        for _ in range(ccount):
            sprite_path = random.choice(sprites)
            try:
                sprite = Image.open(sprite_path).convert("RGBA")
            except Exception as e:
                print(f"Could not open sprite {sprite_path}: {e}")
                continue

            # 1) random width
            target_w = random.randint(MIN_SPRITE_W, MAX_SPRITE_W)
            sprite_resized = resize_sprite_keep_aspect(sprite, target_w)

            # 2) random rotation + crop transparent border
            sprite_final = apply_random_rotation_and_crop(sprite_resized, ROTATION_MAX_DEG)
            if sprite_final is None:
                continue

            # 3) random horizontal flip
            if random.random() < FLIP_PROB:
                sprite_final = ImageOps.mirror(sprite_final)

            sw, sh = sprite_final.size

            # 4) try to place it without overlap
            candidate = place_sprite(bg_w, bg_h, sw, sh, placed_boxes, GAP_PX)
            if candidate is None:
                continue

            xmin, ymin, xmax, ymax = candidate
            # paste preserving alpha
            try:
                bg.paste(sprite_final, (xmin, ymin), sprite_final)
            except Exception:
                bg.paste(sprite_final.convert("RGBA"), (xmin, ymin), sprite_final.split()[-1])

            placed_boxes.append((xmin, ymin, xmax, ymax))
            class_id = CLASS_NAME_TO_ID[lname]
            class_name = ID_TO_NAME[class_id]
            annotations.append((class_id, class_name, xmin, ymin, xmax, ymax))

    # Save composite (PNG)
    out_file = os.path.join(out_path, f"{image_id}.png")
    bg.convert("RGB").save(out_file, "PNG", compress_level=1)
    return out_file, annotations

def build_coco(all_results, classes_map, bg_w, bg_h):
    """
    Build COCO-format JSON dict from all_results:
      all_results: [(image_filename, [(class_id, class_name,xmin,ymin,xmax,ymax), ...]), ...]
      classes_map: dict lower_name->id (CLASS_NAME_TO_ID)
    """
    # categories: use exact provided mapping & canonical names
    categories = []
    for lname, cid in sorted(classes_map.items(), key=lambda x: x[1]):  # sort by id
        categories.append({
            "id": cid,
            "name": ID_TO_NAME[cid],
            "supercategory": "pokemon"
        })

    images = []
    annotations = []
    ann_id = 1
    for img_id, (filename, ann_list) in enumerate(all_results, start=1):
        images.append({
            "id": img_id,
            "file_name": filename,
            "width": bg_w,
            "height": bg_h
        })
        for obj in ann_list:
            class_id, class_name, xmin, ymin, xmax, ymax = obj
            w = xmax - xmin
            h = ymax - ymin
            bbox = [int(xmin), int(ymin), int(w), int(h)]
            area = int(w * h)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(class_id),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": []  # empty because we don't produce polygons
            })
            ann_id += 1

    coco = {
        "info": {
            "description": "Pokemon composite dataset",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return coco

def main():
    # load only the expected class folders (case-insensitive lower names)
    allowed = list(CLASS_NAME_TO_ID.keys())
    classes_dict = load_pokemon_library(pokemon_root, allowed)
    if not classes_dict:
        print("No pokemon classes found under", pokemon_root)
        return

    # sort bg files for deterministic ids
    bg_files = [os.path.join(background_dir, f) for f in os.listdir(background_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))]
    bg_files = bg_files[:300]
    bg_files.sort()
    if not bg_files:
        print("No background images found in", background_dir)
        return

    all_results = []  # list of tuples (image_filename, annotations_list)

    print(f"Found {len(bg_files)} backgrounds and {len(classes_dict)} pokemon classes (from mapping).")
    for idx, bg_path in enumerate(bg_files):
        image_id = f"img_{idx:05d}"
        out_file, annotations = compose_for_background(bg_path, classes_dict, save_dir, image_id)
        if out_file is None:
            continue
        all_results.append((os.path.basename(out_file), annotations))

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1} / {len(bg_files)} backgrounds")

    # Write CSV (same tabular format as before) with canonical names
    max_count = max((len(a) for _, a in all_results), default=0)
    header = ["image_id"]
    for i in range(1, max_count + 1):
        header += [f"pokemon_name_{i}", f"xmin_{i}", f"ymin_{i}", f"xmax_{i}", f"ymax_{i}"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for image_name, annotations in all_results:
            row = [image_name]
            for (class_id, class_name, xmin, ymin, xmax, ymax) in annotations:
                row += [class_name, xmin, ymin, xmax, ymax]
            needed = (max_count - len(annotations)) * 5
            if needed > 0:
                row += [""] * needed
            writer.writerow(row)
    print(f"CSV annotations saved to {csv_path}")

    # Build and write COCO JSON
    coco = build_coco(all_results, CLASS_NAME_TO_ID, BG_SIZE[0], BG_SIZE[1])
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    print(f"COCO JSON saved to {coco_json_path}")

    print(f"Done. Composites saved to {save_dir}.")

if __name__ == "__main__":
    main()
