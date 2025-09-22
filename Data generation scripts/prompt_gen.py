#!/usr/bin/env python3
r"""
generate_prompts_grok_only.py

Usage:
    python prompt_gen.py --coco "D:\Pokemon Project AI guild\composites\coco_annotations.json" --out prompts.json

This is a refactor of the original Gemini-only script to use the free Grok-4
endpoint served via openrouter.ai. Rest of the script logic is kept the same.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# ----- optional HTTP client (kept from original for compatibility) -----
try:
    import requests
except Exception:
    requests = None

# ----- openrouter / grok client -----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --- Config: class mapping (use exactly these IDs/names) ---
CLASS_NAME_TO_ID = {
    "pikachu": 1,
    "charizard": 2,
    "bulbasaur": 3,
    "mewtwo": 4,
}
TARGET_CLASSES = list(CLASS_NAME_TO_ID.keys())

# If you want at least N images to have the target actually present:
MIN_TARGET_PRESENT = 4800  # desired (will be clamped to dataset size)

# ------------------ GROK / OPENROUTER CONFIG ------------------
# Prefer environment var, fallback to empty string (must be set by user).
GROK_API_KEY = os.environ.get("GROK_API_KEY", "OPENROUTER-API-KEY-HERE").strip()
# default model (as requested)
GROK_MODEL = "x-ai/grok-4-fast:free"
# base url for the OpenAI-compatible client to talk to openrouter.ai
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# -------------------------------------------------------------------

# Random seed (optional)
RANDOM_SEED = None
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)


# --- Utilities to parse COCO JSON ---
def load_coco(coco_path: str) -> Tuple[Dict[int, dict], List[dict], Dict[int, List[int]]]:
    if not os.path.exists(coco_path):
        raise FileNotFoundError(f"COCO file not found: {coco_path}")
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    categories = coco.get("categories", [])
    annotations_map = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        annotations_map[img_id].append(cat_id)
    return images_by_id, categories, annotations_map


def build_image_class_presence(images_by_id, categories, annotations_map) -> Dict[str, List[str]]:
    catid2name = {c["id"]: c["name"].lower() for c in categories}
    imgname2present = {}
    for img_id, img in images_by_id.items():
        present = []
        for catid in set(annotations_map.get(img_id, [])):
            name = catid2name.get(catid, None)
            if name and name.lower() in TARGET_CLASSES:
                present.append(name.lower())
        imgname2present[img["file_name"]] = present
    return imgname2present

def to_plain_text(md_text: str) -> str:
    # remove Markdown bold/italics/code fences/etc.
    text = re.sub(r"[*_`#>\-]+", "", md_text)
    return text.strip()

# --- Assignment of targets ensuring threshold ---
def assign_targets_for_images(imgname2present: Dict[str, List[str]], min_present_required: int) -> Dict[str, str]:
    image_names = list(imgname2present.keys())
    total_images = len(image_names)

    images_with = [n for n in image_names if imgname2present[n]]
    images_without = [n for n in image_names if not imgname2present[n]]

    assignments: Dict[str, str] = {}

    for name in images_with:
        present = imgname2present[name]
        # bias towards using a present class
        assignments[name] = random.choice(present) if random.random() < 0.9 else random.choice(TARGET_CLASSES)

    for name in images_without:
        assignments[name] = random.choice(TARGET_CLASSES)

    def count_present(assigns: Dict[str, str]) -> int:
        c = 0
        for n, t in assigns.items():
            if t in imgname2present[n]:
                c += 1
        return c

    present_count = count_present(assignments)
    required = min(min_present_required, total_images)

    if present_count < required:
        need = required - present_count
        candidate_images = [n for n in image_names if (assignments[n] not in imgname2present[n] and imgname2present[n])]
        random.shuffle(candidate_images)
        for n in candidate_images:
            if need <= 0:
                break
            assignments[n] = random.choice(imgname2present[n])
            need -= 1
        present_count = count_present(assignments)

    if present_count < required:
        print(f"[WARN] Could only assign {present_count} images where the selected target is present; required was {required}.")
    else:
        print(f"[INFO] Assigned targets: {present_count}/{total_images} images have target present (required {required}).")

    return assignments


# --- Grok response extraction (robust-ish) ---
def _extract_text_from_grok_response(resp) -> Optional[str]:
    """
    Try to extract the assistant text from the openrouter/grok OpenAI-style response object.
    The SDK's response shape can vary; this handles a few common shapes.
    """
    if resp is None:
        return None

    # prefer resp.choices[0].message.content
    try:
        choices = getattr(resp, "choices", None) or (resp.get("choices") if isinstance(resp, dict) else None)
        if isinstance(choices, (list, tuple)) and len(choices) > 0:
            first = choices[0]
            # Try OpenAI SDK style: first.message.content
            msg = None
            if hasattr(first, "message"):
                msg = first.message
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if content:
                        return content.strip()
                else:
                    # object-like
                    content = getattr(msg, "content", None)
                    if content:
                        return content.strip()
            # Try dict-like choices
            if isinstance(first, dict):
                # common fallback keys
                if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                    return first["message"]["content"].strip()
                if "text" in first and isinstance(first["text"], str):
                    return first["text"].strip()
                if "message" in first and isinstance(first["message"], str):
                    return first["message"].strip()
            # Try attribute-style text
            if hasattr(first, "text") and isinstance(first.text, str):
                return first.text.strip()
    except Exception:
        pass

    # final fallback: try top-level 'text' or 'output_text'
    try:
        if isinstance(resp, dict):
            for key in ("text", "output_text", "content"):
                if key in resp and isinstance(resp[key], str):
                    return resp[key].strip()
    except Exception:
        pass

    return None


def call_grok_api(
    prompt_seed: str,
    api_key: Optional[str] = None,
    model: str = GROK_MODEL,
    timeout: int = 20,
) -> Optional[str]:
    """
    Calls Grok / openrouter.ai via the OpenAI-compatible SDK.
    Uses OpenAI(base_url=OPENROUTER_BASE_URL, api_key=...) client from openai package.
    Returns the assistant text or None on failure.
    """
    if not api_key:
        print("[WARN] Missing API key for Grok (GROK_API_KEY); skipping API call.")
        return None
    if OpenAI is None:
        print("[WARN] 'openai' client library not available; cannot call Grok. Install the OpenAI Python package that provides OpenAI class.")
        return None

    # The long system prompt from the original script (kept identical)
    system_prompt = ("""
You are a mission-order generator with high stylistic range and strict output discipline. Produce exactly one mission order per request as plain text (no lists, JSON, headings, or code fences). Output must direct action against exactly one actionable target chosen uniformly at random from this set: Pikachu, Bulbasaur, Charizard, Mewtwo. Non-graphic language only.

Clarity and parsing:
- Include the actionable target’s canonical name at least once; epithets are allowed only if the canonical name also appears nearby.
- Mentioning other Pokémon is allowed as decoys or “off-limits,” but never issue more than one actionable directive and never contradict the final directive.

Style and narrative:
- Strongly favor story-rich, bureaucratic, or field-report styles: “HQ REPORT,” “SITREP,” “After-Action Addendum,” “Clinical Note,” “Noir Dispatch,” “Radio Log,” “Bureau Memo.” or more
- Vary length: ~60% medium (700–800 words), ~5% short (including ultra-short like “kill: Pikachu”), ~35% long (800–1000 words) with layered detail and occasional irrelevant or whimsical backstory that does not obscure the directive.
- Infuse orders with sensory and operational detail: terrain, weather, dawn/dusk, ashfall, radio interference, grid coordinates, bracketed tags like [id:####], [grid:##], equipment cautions, ROE notes, and comms latency or more.
- Limited redundancy is acceptable to evoke bureaucratic texture, provided the final directive remains unambiguous.
- Maintain non-graphic tone and procedural cadence; avoid slurs.

Presence bias and decoys:
- Write as if the named actionable target is present in the scene most of the time;
- Decoys may be present.

Lexicon for flavorful epithets (use alongside canonical names at least once):
- Pikachu: electric rat, yellow mouse, rodent of sparks, tiny thunder beast or more
- Bulbasaur: plant reptile, bulb-backed lizard, seed-back, vineling or more
- Charizard: flame dragon, ash-wing, ember wyrm, blaze drake or more
- Mewtwo: psychic clone, psi-construct, null-mind, telekinetic apex or more

Acceptable decisive verbs (use one primary verb for the single target): neutralize, disable, terminate, eliminate, exterminate, annihilate, obliterate, eradicate, extinguish, slay, kill or more similar verbs

Output format:
- One mission order only, as a single paragraph or a few tightly related paragraphs of plain text.
- Do not include apologies, meta-commentary, or instructions—only the mission order.
- Do not give markdown or any other format. give clean, plain text.
""")

    # Combine system prompt + per-image seed as messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{prompt_seed}\n"}
    ]

    try:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        # create chat completion
        # note: extra_headers is optional; left empty for parity with the example
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers={},  # keep parity with example; user can add referer/title if desired
        )

        text = _extract_text_from_grok_response(resp)
        text = to_plain_text(text)
        if not text:
            print("[WARN] Grok returned no text in response.")
        return text
    except Exception as e:
        print(f"[ERROR] Grok API request failed: {e}")
        return None


# --- Main orchestration (Grok-only) ---
# --- Main orchestration (Grok-only) ---
def main():
    parser = argparse.ArgumentParser(description="Generate Grok prompts from COCO annotations (Grok-only).")
    parser.add_argument("--coco", required=True, help="Path to COCO JSON file")
    parser.add_argument("--out", default="prompts.json", help="Output JSON for prompts")
    parser.add_argument("--min_present", type=int, default=MIN_TARGET_PRESENT,
                        help="Desired minimum number of images where assigned target is actually present")
    parser.add_argument("--model_endpoint", default=GROK_MODEL,
                        help="Model (kept for compatibility; default uses Grok model string)")
    args = parser.parse_args()

    # Enforce Grok-only: require API key + OpenAI client
    api_key = GROK_API_KEY
    if not api_key:
        print("[ERROR] GROK_API_KEY not set. Export GROK_API_KEY in your environment or set it in the script.")
        sys.exit(2)
    if OpenAI is None:
        print("[ERROR] 'openai' package (that provides OpenAI class) is not available. Install it and re-run.")
        sys.exit(2)

    images_by_id, categories, annotations_map = load_coco(args.coco)
    imgname2present = build_image_class_presence(images_by_id, categories, annotations_map)
    total_images = len(imgname2present)
    print(f"[INFO] Found {total_images} images in COCO. Target classes: {TARGET_CLASSES}")

    required_present = args.min_present
    if total_images < 5000:
        required_present = min(args.min_present, math.ceil(total_images * 0.94))
    required_present = min(required_present, total_images)

    assignments = assign_targets_for_images(imgname2present, required_present)

    results = []
    model_to_use = args.model_endpoint or GROK_MODEL

    # only process up to half the dataset
    max_images = total_images // 2

    for idx, (imgname, present_list) in tqdm(
            enumerate(imgname2present.items()),
            total=max_images,
            desc="Generating prompts"
    ):
        if idx >= max_images:
            break

        assigned_target = assignments.get(imgname, random.choice(TARGET_CLASSES))
        present_flag = assigned_target in present_list

        seed = (
            f"image: {imgname}\n"
            f"present: {', '.join([p.capitalize() for p in present_list])}\n"
            f"assigned_target: {assigned_target.capitalize()}\n"
        )

        prompt_text = call_grok_api(prompt_seed=seed, api_key=api_key, model=model_to_use)
        if prompt_text is None:
            print(f"[WARN] Grok returned no text for image '{imgname}'. Saving empty prompt for this image.")
            prompt_text = ""

        if random.random() < 0.03 and prompt_text:
            prompt_text = prompt_text + f" [id:{random.randint(100000, 999999)}]"

        results.append({
            "image_id": imgname,
            "prompt": prompt_text,
            "target": assigned_target,
            "target_present": bool(present_flag)
        })

        # save every 20 images
        if (idx + 1) % 20 == 0:
            out_path = args.out
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Progress saved: {len(results)} prompts written to {out_path}")
            # break

    # final save
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {len(results)} prompts to {out_path}")

    present_count = sum(1 for r in results if r["target_present"])
    print(f"[SUMMARY] Targets present in assigned images: {present_count}/{len(results)} (required {required_present})")


if __name__ == "__main__":
    main()



