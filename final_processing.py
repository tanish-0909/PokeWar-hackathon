from pathlib import Path
import json
from typing import Optional

import pandas as pd


def _normalize_class_id(value) -> str:
    """Normalize class id so it's a string in '1'..'4' when possible.

    If value is numeric and in 0..3 it becomes 1..4 by adding 1.
    If value is numeric but outside that range we return its integer string.
    Otherwise we return the stripped string representation.
    """
    s = str(value).strip()
    try:
        # handle values like '1', 1.0, '1.0' etc.
        i = int(float(s))
    except Exception:
        return s
    if 0 <= i <= 3:
        return str(i + 1)
    return str(i)


def extract_centers(
    csv_path: str,
    json_in_path: str,
    csv_out_path: Optional[str] = None,
) -> str:
    """Read a CSV and JSON file and write a CSV of center points per prediction.

    Parameters
    ----------
    csv_path : str
        Path to the CSV containing detected Pokemon. Must contain the columns
        ['image_id', 'pokemon_id', 'x_center', 'y_center'].
    json_in_path : str
        Path to the JSON file containing predictions. Each element is expected
        to be a mapping that contains at least 'image_id' and 'pred_class_id'.
    csv_out_path : Optional[str], optional
        Where to save the resulting CSV. If None, a file next to `json_in_path`
        with suffix "_centers.csv" will be created.

    Returns
    -------
    str
        Absolute path to the saved CSV file.

    Raises
    ------
    ValueError
        If the CSV is missing required columns.
    """

    csv_path = str(csv_path)
    json_in_path = str(json_in_path)

    # determine output path
    if csv_out_path is None:
        inp = Path(json_in_path)
        csv_out_path = str(inp.with_name(inp.stem + "_centers.csv"))

    # read dataframe
    df = pd.read_csv(csv_path)
    print(f"detected {len(df)} number of pokemon (rows in dataframe)")

    # basic column checks
    required_cols = {"image_id", "pokemon_id", "x_center", "y_center"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataframe missing required columns. Found: {df.columns.tolist()}")

    # normalize types to strings for safe matching
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)

    # create pokemon_class from pokemon_id and normalize class ids to 1..4 if needed
    df["pokemon_class"] = df["pokemon_id"].apply(_normalize_class_id)

    # read json
    with open(json_in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # sort (optional) for consistent order
    data = sorted(data, key=lambda x: (str(x.get("image_id", "")), str(x.get("pred_class_id", ""))))

    rows = []

    for point in data:
        img = str(point.get("image_id", "")).strip()
        # normalize pred_class_id the same way as CSV side so they sync up
        target = _normalize_class_id(point.get("pred_class_id", ""))

        if not img:
            continue  # skip malformed entries

        # select matching rows
        matches = df[(df["image_id"] == img) & (df["pokemon_class"] == target)]

        # collect centers
        centers = matches[["x_center", "y_center"]].apply(
            lambda row: [float(row["x_center"]), float(row["y_center"])], axis=1
        ).tolist()

        # add row for CSV (store centers as string to match original script)
        rows.append({"image_id": img, "points": str(centers)})

    # convert to DataFrame and save to CSV
    out_df = pd.DataFrame(rows)

    out_path = Path(csv_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"CSV written to: {out_path}")
    print("Example rows:")
    print(out_df.head())

    return str(out_path.resolve())


if __name__ == "__main__":
    example_csv = r"D:\Pokemon Project AI guild\predictions.csv"
    example_json = r"D:\Pokemon Project AI guild\google_big_bird_pred2.json"
    example_out = None

    try:
        saved = extract_centers(example_csv, example_json, example_out)
        print('Saved to:', saved)
    except Exception as e:
        print('Error:', e)
