import os
from PIL import Image

image_dir = r"D:\Pokemon Project AI guild\background images"
save_dir = r"D:\Pokemon Project AI guild\cropped_bg_images"
os.makedirs(save_dir, exist_ok=True)

image_size = (640, 480)  # width, height

count = 0
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)

    try:
        with Image.open(img_path) as img:
            # Convert to RGB (avoids issues with PNG / alpha channels)
            img = img.convert("RGB")
            # Resize to (640x480)
            img_resized = img.resize(image_size, Image.Resampling.LANCZOS)

            save_path = os.path.join(
                save_dir, f"{os.path.splitext(img_name)[0]}_resized.jpg"
            )
            img_resized.save(save_path, "JPEG")

            count += 1
            if count % 100 == 0:
                print(f"Saved {count} images to {save_dir}")

    except Exception as e:
        print(f"❌ Could not process {img_name}: {e}")

print(f"✅ Finished. Saved {count} images to {save_dir}")
