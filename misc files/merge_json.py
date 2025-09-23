# #!/usr/bin/env python3
# """
# merge_prompts.py
#
# Usage:
#     python merge_json.py prompts.json prompts_last2500.json -o combined.json
#
# Merges multiple JSON files containing lists of records like:
#   { "image_id": "img_00082.png", "prompt": "...", "target": "bulbasaur", "target_present": true }
#
# For records with empty prompt ("" or whitespace), replaces prompt with "kill: {target}".
# If the same image_id appears in multiple files, the script keeps the record with a non-empty prompt
# (if available), otherwise the first seen record is kept.
#
# Output is a JSON list sorted by the numeric part of image_id where possible.
# """
# import argparse
# import json
# import re
# from typing import Any, Dict, List
#
#
# def load_records(path: str) -> List[Dict[str, Any]]:
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     if isinstance(data, list):
#         return data
#     # if file contains a single object or mapping, attempt to convert to list
#     if isinstance(data, dict):
#         # if dict keyed by image_id -> record
#         # convert to list of values (best-effort)
#         return list(data.values())
#     raise ValueError(f"Unsupported JSON root type in {path}: {type(data)}")
#
#
# def is_empty_prompt(p: Any) -> bool:
#     return (p is None) or (isinstance(p, str) and p.strip() == "")
#
#
# def natural_image_sort_key(image_id: str):
#     # find first (or best) digit group and use it as integer key; fallback to full string
#     m = re.search(r"(\d+)", image_id)
#     if m:
#         return (0, int(m.group(1)), image_id)  # prefix 0 so numeric-sorted items come before non-numeric
#     return (1, 0, image_id)
#
#
# def merge_records(file_paths: List[str]) -> List[Dict[str, Any]]:
#     merged: Dict[str, Dict[str, Any]] = {}
#     for path in file_paths:
#         records = load_records(path)
#         for rec in records:
#             if not isinstance(rec, dict):
#                 continue
#             imgid = rec.get("image_id")
#             if not imgid:
#                 # skip records without image_id
#                 continue
#             # If this image_id already present, prefer a non-empty prompt record
#             existing = merged.get(imgid)
#             if existing is None:
#                 merged[imgid] = rec.copy()
#             else:
#                 # if existing prompt empty and new one non-empty => replace
#                 if is_empty_prompt(existing.get("prompt")) and not is_empty_prompt(rec.get("prompt")):
#                     merged[imgid] = rec.copy()
#                 # otherwise keep existing (first seen) to preserve deterministic behavior
#     # After merging, fill empty prompts
#     filled_count = 0
#     for imgid, rec in merged.items():
#         if is_empty_prompt(rec.get("prompt")):
#             target = rec.get("target") or rec.get("class") or "unknown"
#             rec["prompt"] = f"kill: {target}"
#             filled_count += 1
#     # sort by natural image id
#     items = list(merged.values())
#     items.sort(key=lambda r: natural_image_sort_key(str(r.get("image_id", ""))))
#     return items
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Merge prompt JSON files and fill empty prompts with kill:{target}")
#     parser.add_argument("inputs", nargs="+", help="Input JSON files (two or more)")
#     parser.add_argument("-o", "--output", default="combined.json", help="Output JSON path")
#     args = parser.parse_args()
#
#     merged = merge_records(args.inputs)
#     with open(args.output, "w", encoding="utf-8") as f:
#         json.dump(merged, f, indent=2, ensure_ascii=False)
#
#     total = len(merged)
#     filled = sum(1 for r in merged if r.get("prompt", "").strip().startswith("kill:"))
#     print(f"[INFO] Wrote {total} records to {args.output}  ({filled} prompts auto-filled with 'kill:')")
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
coco_replicate_entries.py

Replicate COCO 'images' entries and duplicate their annotations for existing augmented image files.

Examples:
    # replicate using generated suffixes _aug1 and _aug2
    python coco_replicate_entries.py --input coco_orig.json --output coco_with_aug.json --suffixes _aug1,_aug2

    # replicate using N augmentations with default suffix template "_aug{idx}"
    python coco_replicate_entries.py --input coco_orig.json --output coco_with_aug.json --n-aug 2

    # replicate only for a subset of images whose file_names match a prefix:
    python coco_replicate_entries.py --input coco_orig.json --output coco_with_aug.json --suffixes _aug1 --filter-prefix img_000

Notes:
 - The script will assign new unique image and annotation ids.
 - It will not change bbox/segmentation; you must ensure the actual augmented images exist and annotations remain valid.
"""
import argparse
import json
import os
import copy
from typing import Dict, List

def insert_suffix_before_ext(filename: str, suffix: str) -> str:
    base, ext = os.path.splitext(filename)
    return f"{base}{suffix}{ext}"

def replicate_entries(
    coco: Dict,
    suffixes: List[str],
    filter_prefix: str = None,
    skip_if_filename_exists: bool = True
) -> Dict:
    """
    For each image in coco['images'] (optionally filtered by prefix), create duplicate image
    entries with file_names modified by each suffix in suffixes, and duplicate annotations
    belonging to the original image (with new unique IDs and image_id mapping).
    """
    images = coco.get('images', [])
    annotations = coco.get('annotations', [])

    if images is None:
        raise ValueError("Input COCO does not contain 'images' key.")

    # Build annotation lookup by image_id
    anns_by_image = {}
    for ann in annotations:
        img_id = int(ann['image_id'])
        anns_by_image.setdefault(img_id, []).append(ann)

    # Determine next available ids
    existing_image_ids = [int(img['id']) for img in images] if images else []
    existing_ann_ids = [int(ann['id']) for ann in annotations] if annotations else []

    next_image_id = max(existing_image_ids) + 1 if existing_image_ids else 1
    next_ann_id = max(existing_ann_ids) + 1 if existing_ann_ids else 1

    # Track filenames to avoid duplicates
    existing_filenames = set(img.get('file_name') for img in images)

    new_images = []
    new_annotations = []

    for img in images:
        img_id = int(img['id'])
        fname = img.get('file_name')
        if fname is None:
            print(f"Skipping image id {img_id}: no file_name")
            continue

        if filter_prefix and not fname.startswith(filter_prefix):
            # skip images that don't match the prefix filter
            continue

        orig_ann_list = anns_by_image.get(img_id, [])

        for suf in suffixes:
            new_fname = insert_suffix_before_ext(fname, suf)

            if skip_if_filename_exists and new_fname in existing_filenames:
                # don't add duplicate filename entry
                print(f"Skipping {new_fname}: filename already exists in dataset.")
                continue

            # create new image entry (deep copy to preserve fields)
            new_img = copy.deepcopy(img)
            new_img['id'] = next_image_id
            new_img['file_name'] = new_fname
            # optional flag to indicate augmented entry
            # new_img['is_augmented'] = True
            new_images.append(new_img)
            existing_filenames.add(new_fname)

            # duplicate annotations for this image
            for ann in orig_ann_list:
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = next_ann_id
                new_ann['image_id'] = next_image_id
                new_annotations.append(new_ann)
                next_ann_id += 1

            next_image_id += 1

    # Append new entries to coco structure
    coco_out = copy.deepcopy(coco)
    coco_out['images'] = images + new_images
    coco_out['annotations'] = annotations + new_annotations

    print(f"Added {len(new_images)} images and {len(new_annotations)} annotations.")
    return coco_out

def parse_suffixes_arg(s: str, n_aug: int = None) -> List[str]:
    if s:
        # allow comma-separated suffix list
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return parts
    if n_aug and n_aug > 0:
        return [f"_aug{i+1}" for i in range(n_aug)]
    return []

def main():
    p = argparse.ArgumentParser(description="Replicate COCO JSON entries for augmented images (modify filenames and duplicate annotations).")
    p.add_argument("--input", "-i", required=True, help="Input COCO JSON file path")
    p.add_argument("--output", "-o", required=True, help="Output COCO JSON file path")
    p.add_argument("--suffixes", type=str, default="", help="Comma-separated list of suffixes to append before extension, e.g. '_aug1,_aug2'")
    p.add_argument("--n-aug", type=int, default=0, help="If suffixes not provided, create N suffixes named _aug1.._augN")
    p.add_argument("--filter-prefix", type=str, default=None, help="Only replicate images whose file_name starts with this prefix (optional)")
    p.add_argument("--skip-if-filename-exists", action="store_true", help="Don't create replicated entry if target filename is already present in dataset")
    args = p.parse_args()

    # Load JSON
    with open(args.input, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    suffixes = parse_suffixes_arg(args.suffixes, args.n_aug)
    if not suffixes:
        p.error("No suffixes provided. Use --suffixes or --n-aug > 0")

    out = replicate_entries(
        coco,
        suffixes=suffixes,
        filter_prefix=args.filter_prefix,
        skip_if_filename_exists=args.skip_if_filename_exists
    )

    # write output
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote augmented COCO JSON to: {args.output}")

if __name__ == "__main__":
    main()
