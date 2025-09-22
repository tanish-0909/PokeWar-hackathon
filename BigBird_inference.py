import os, json
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

def run_bigbird_inference(model_dir: str, new_json: str, output_csv: str, batch_size: int = 8):
    """
    Runs inference using a fine-tuned BigBird (or any HuggingFace transformer) model.

    Args:
        model_dir (str): Path to trained model directory (contains `meta.json` and model files).
        new_json (str): Path to JSON input file (list of dicts with `prompt` or text field).
        output_csv (str): Path to save predictions CSV.
        batch_size (int): Inference batch size.

    Returns:
        str: Path to saved CSV file (output_csv).
    """

    # -------------------- Load meta --------------------
    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    id_to_class = {int(k): v for k,v in meta["class_id_to_name"].items()}
    max_length = meta.get("max_length", 512)

    # -------------------- Load model + tokenizer --------------------
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # -------------------- Load new JSON --------------------
    def detect_label_column(obj):
        possible = ['label', 'class', 'class_id', 'target']
        for p in possible:
            if p in obj: return p
        for p in obj.keys():
            if p.lower() in possible: return p
        return None

    def load_json_dataset(path):
        encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
        data, last_exc = None, None
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    data = json.load(f)
                break
            except Exception as e:
                last_exc = e
        if data is None:
            raise last_exc

        if isinstance(data, dict):
            for k in ['data','items','examples','instances']:
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break

        if not isinstance(data, list):
            raise ValueError("JSON must be a list (or dict with list under 'data/items').")

        rows = []
        for obj in data:
            image_id = obj.get('image_id') or obj.get('id') or obj.get('img_id') or obj.get('image')
            prompt = obj.get('prompt') or obj.get('text') or obj.get('order') or obj.get('instruction')
            if prompt is None:
                strings = [(k,v) for k,v in obj.items() if isinstance(v,str)]
                if strings:
                    prompt = max(strings, key=lambda t: len(t[1]))[1]
            rows.append({'image_id': image_id, 'prompt': prompt})
        return pd.DataFrame(rows)

    df_new = load_json_dataset(new_json)

    # -------------------- Predict --------------------
    pred_labels, pred_probs = [], []
    for i in range(0, len(df_new), batch_size):
        batch_prompts = df_new['prompt'].iloc[i:i+batch_size].fillna("").tolist()
        enc = tokenizer(batch_prompts, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        pred_labels.extend(preds)
        pred_probs.extend(probs)

    df_new["pred_class_id"] = [int(x) for x in pred_labels]
    df_new["pred_class_name"] = [id_to_class[x+1] for x in pred_labels]  # adjust offset
    df_new["probs"] = pred_probs

    df_out = df_new[["image_id", "prompt", "pred_class_id", "pred_class_name", "probs"]]
    df_out.to_csv(output_csv, index=False)

    return output_csv
