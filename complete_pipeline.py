"""
    End to end pipeline.
    Inputs: json file with HQ orders, respective test image directory
    Outputs: final submission csv, final images marked with X for each kill

"""

import json
import pandas as pd

from BigBird_inference import run_bigbird_inference
from yolo_inference import run_yolo_inference
from final_processing import extract_centers

test_dir = r"D:\Pokemon Project AI guild\test_images"
test_json = r"D:\Pokemon Project AI guild\test_prompts_orders.json"

big_bird_inference_dir = r""
yolo_weights = ""

with open(test_json, "r", encoding="utf-8") as f:
    test_prompts = json.load(f)

yolo_out_csv_path = run_yolo_inference(yolo_weights, test_dir, "D:\Pokemon Project AI guild\yolo_output.csv")
bb_output_csv_path = run_bigbird_inference(big_bird_inference_dir, test_json, OUTPUT_CSV := r"D:\Pokemon Project AI guild\bb_output.csv")

import pandas as pd

OUTPUT_JSON = r"D:\Pokemon Project AI guild\google_big_bird_pred.json"
df_out = pd.read_csv(OUTPUT_CSV)

# Convert to JSON (records format is convenient for lists of objects)
df_out.to_json(OUTPUT_JSON, orient="records", indent=2)

output_csv = extract_centers(yolo_out_csv_path, OUTPUT_JSON)

print(f"The pipeline has completed processing and the final result has been stored in {output_csv}")
