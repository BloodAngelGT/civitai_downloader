#!/usr/bin/env python3

import requests
import os
import re
import sys
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CIVITAI_API_KEY")
COMFYUI_MODEL_ROOT = os.getenv("COMFYUI_MODEL_ROOT", "/mnt/user/appdata/comfyui-nvidia/mnt/ComfyUI/models")
CIVITAI_BASE_API = "https://civitai.com/api/v1/model-versions"

if not API_KEY:
    print("Fehler: API-Key nicht gefunden. Lege eine .env-Datei mit 'CIVITAI_API_KEY=...' an.")
    sys.exit(1)

def extract_model_version_id(url):
    match = re.search(r'modelVersionId=(\d+)', url)
    if match:
        return match.group(1)
    print("Fehler: Kein 'modelVersionId' gefunden. Bitte vollständige Modell-URL verwenden.")
    sys.exit(1)

def get_model_metadata(model_version_id):
    url = f"{CIVITAI_BASE_API}/{model_version_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_download_info(metadata):
    download_url = f"https://civitai.com/api/download/models/{metadata['id']}?type=Model&format=SafeTensor"
    base_model = metadata.get("baseModel", "unknown").replace(" ", "")
    model_type = metadata.get("model", {}).get("type", "Unknown")
    filename = metadata["files"][0]["name"]

    raw_tags = metadata.get("model", {}).get("tags", [])
    tags = []
    for tag in raw_tags:
        if isinstance(tag, str):
            tags.append(tag.lower())
        elif isinstance(tag, dict) and "name" in tag:
            tags.append(tag["name"].lower())

    return download_url, base_model, model_type, filename, tags

def comfyui_path_for(model_type):
    model_type = model_type.lower()
    return {
        "checkpoint": "checkpoints",
        "locon": "loras",
        "lora": "loras",
        "textualinversion": "embeddings",
        "hypernetwork": "hypernetworks"
    }.get(model_type, "other")

def infer_subfolder(tags):
    if "nsfw" in tags:
        return "NSFW"
    if "experimental" in tags or "beta" in tags:
        return "experimental"
    return ""

def download_file(url, output_path, api_key):
    if os.path.exists(output_path):
        print(f"Datei existiert bereits, überspringe: {output_path}")
        return

    headers = {"Authorization": f"Bearer {api_key}"}
    with requests.get(url + f"&token={api_key}", headers=headers, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(output_path)}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

def main():
    if len(sys.argv) != 2:
        print("Verwendung: ./civitai_downloader.py \"https://civitai.com/...modelVersionId=XXXX\"")
        sys.exit(1)

    civitai_url = sys.argv[1].strip()
    model_version_id = extract_model_version_id(civitai_url)

    print(f"Model-Version-ID: {model_version_id} wird verarbeitet...")

    metadata = get_model_metadata(model_version_id)
    download_url, base_model, model_type, filename, tags = get_download_info(metadata)

    comfy_subdir = comfyui_path_for(model_type)
    subfolder = infer_subfolder(tags)

    ziel_verzeichnis = os.path.join(COMFYUI_MODEL_ROOT, comfy_subdir, base_model)
    if subfolder:
        ziel_verzeichnis = os.path.join(ziel_verzeichnis, subfolder)

    ziel_pfad = os.path.join(ziel_verzeichnis, filename)

    print(f"Modell-Typ: {model_type}")
    print(f"Base Model: {base_model}")
    print(f"Tags: {tags}")
    print(f"Speicherort: {ziel_pfad}")

    download_file(download_url, ziel_pfad, API_KEY)
    print("\nDownload abgeschlossen.")

if __name__ == "__main__":
    main()