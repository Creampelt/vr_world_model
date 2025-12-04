import json
import os
from pathlib import Path
import uuid
import requests
import logging
from tqdm import tqdm, trange

MY_DIR = Path(__file__).parent
PROMPT_JSON = MY_DIR / "prompts.json"
BATCH_SIZE = 5
API_URL = "http://0.0.0.0:8000/generate"

DEFAULT_NEGATIVE_PROMPT = """
Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, image, still,
overall grayish, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers,
poorly drawn hands, poorly drawn faces, deformed, disfigured, distorted limbs, fingers fused together,
static image, cluttered background, three legs, people, walking backwards.
"""
DEFAULT_NEGATIVE_PROMPT = " ".join(
    line.strip() for line in DEFAULT_NEGATIVE_PROMPT.strip().splitlines()
)

OUTPUT_PATH = MY_DIR / "results.json"

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def main():
    prompts = load_json(PROMPT_JSON)
    results = {}

    for i in trange(0, len(prompts), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(prompts))
        batch = prompts[i:end]

        prompts_json = {
            "positive_prompts": [p["prompt"] for p in batch],
            "negative_prompts": [DEFAULT_NEGATIVE_PROMPT] * len(batch),
        }
        files = [("prompts_json", (None, json.dumps(prompts_json)))]
        for p in batch:
            files.append(("images", open(p["image_path"], "rb")))

        response = requests.post(API_URL, files=files)
        logging.info(response.text)

        response_res = json.loads(response.text)
        new_entries = {}

        for p, res in zip(batch, response_res):
            new_entries[p["id"]] = {"input": p, "output": res}

        results.update(new_entries)
        save_json(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
