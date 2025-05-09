#!/usr/bin/env python3

import requests
import os
import re
import sys
from tqdm import tqdm
from dotenv import load_dotenv

# === .env laden ===
load_dotenv()
API_KEY = os.getenv("CIVITAI_API_KEY")
COMFYUI_MODEL_ROOT = os.getenv("COMFYUI_MODEL_ROOT", "/mnt/user/appdata/comfyui-nvidia/mnt/ComfyUI/models")
CIVITAI_BASE_API = "https://civitai.com/api/v1/models"

if not API_KEY:
    print("Fehler: API-Key nicht gefunden. Lege eine .env-Datei mit 'CIVITAI_API_KEY=...' an.")
    sys.exit(1)

def extract_model_id(url):
    match = re.search(r'/models/(\d+)', url)
    return match.group(1) if match else None

def get_model_metadata(model_id):
    url = f"{CIVITAI_BASE_API}/{model_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_download_info(metadata):
    version = metadata["modelVersions"][0]
    download_url = version["downloadUrl"]
    base_model = version.get("baseModel", "unknown").replace(" ", "")
    model_type = metadata.get("type", "Unknown")
    filename = version["files"][0]["name"]
    tags = [t['name'].lower() for t in metadata.get("tags", [])]
    return download_url, base_model, model_type, filename, tags

def comfyui_path_for(model_type):
    return {
        "Checkpoint": "checkpoints",
        "LoCon": "loras",
        "LoRA": "loras",
        "TextualInversion": "embeddings",
        "Hypernetwork": "hypernetworks"
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
        print("Verwendung: civitai_downloader <Civitai-Link>")
        sys.exit(1)

    civitai_url = sys.argv[1].strip()
    model_id = extract_model_id(civitai_url)
    if not model_id:
        print("Ungültige URL. Beispiel: https://civitai.com/models/123456")
        sys.exit(1)

    print(f"Modell-ID: {model_id} wird verarbeitet...")

    metadata = get_model_metadata(model_id)
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
