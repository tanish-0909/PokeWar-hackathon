import pandas as pd
import ast
from pathlib import Path
from PIL import Image, ImageDraw

# === Paths (adjust as needed) ===
csv_path = r"D:\Pokemon Project AI guild\google_big_bird_pred_centers.csv"
images_dir = r"D:\Pokemon Project AI guild\test_images"   # folder where your images are stored
output_dir = r"D:\Pokemon Project AI guild\plotted_images"

# === Load CSV ===
df = pd.read_csv(csv_path)

# convert "points" from string to actual list of lists
df["points"] = df["points"].apply(ast.literal_eval)

# ensure output folder exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# === Plot each image with X marks using PIL ===
for _, row in df.iterrows():
    img_id = row["image_id"]
    points = row["points"]

    img_path = Path(images_dir) / img_id
    if not img_path.exists():
        print(f"Warning: image {img_path} not found, skipping...")
        continue

    # open image
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # draw X markers
    for (x, y) in points:
        size = 10  # half-size of the X arms
        color = "red"
        thickness = 3

        # draw the two crossing lines of the X
        draw.line((x - size, y - size, x + size, y + size), fill=color, width=thickness)
        draw.line((x - size, y + size, x + size, y - size), fill=color, width=thickness)

    # save output
    out_path = Path(output_dir) / img_id
    img.save(out_path)
    print(f"Saved plotted image with centers: {out_path}")
