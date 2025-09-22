import os, glob
import pandas as pd
from ultralytics import YOLO

def run_yolo_inference(
    weights_path: str,
    img_folder: str,
    output_csv: str,
    conf: float = 0.5
):
    """
    Run YOLO object detection inference on a folder of images.

    Args:
        weights_path (str): Path to trained YOLO weights (best.pt).
        img_folder (str): Path to folder containing test images (.jpg/.png).
        output_csv (str): Path to save predictions CSV.
        conf (float): Confidence threshold for YOLO predictions.

    Returns:
        str: Path to saved CSV file (output_csv).
    """
    # Load YOLO model
    model = YOLO(weights_path)

    # Collect image paths
    img_paths = glob.glob(os.path.join(img_folder, "*.png")) + \
                glob.glob(os.path.join(img_folder, "*.jpg"))

    rows = []
    for img_path in img_paths:
        results = model.predict(img_path, conf=conf, save=False, verbose=False)
        res = results[0]

        image_id = os.path.basename(img_path)

        # Iterate detections
        for box in res.boxes:
            cls_id = int(box.cls.item())
            x_center, y_center, width, height = box.xywh[0].tolist()

            rows.append({
                "image_id": image_id,
                "pokemon_id": cls_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    return output_csv
