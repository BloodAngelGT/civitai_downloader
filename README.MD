# Civitai Downloader für ComfyUI

Dieses Python-Skript lädt automatisch Modelle von [Civitai](https://civitai.com) anhand ihrer `modelVersionId` herunter, inklusive Authentifizierung über API-Key. Die heruntergeladenen Dateien werden automatisch in die passende Ordnerstruktur für **ComfyUI** einsortiert (z. B. `loras/SDXL1.0/NSFW`).

## Funktionen

- Interaktive oder Parameter-basierte Eingabe der URL
- Automatische Zuordnung zu LoRA, Checkpoints, Embeddings etc.
- Unterstützt NSFW-Tag-Ordner
- Erkennt das BaseModel (z. B. `SDXL1.0`) automatisch
- Erneute Eingabe nach jedem Download möglich

## Voraussetzungen

- Python 3.9+
- `.env`-Datei mit API-Key und Zielpfad (Beispiel unten)
- Abhängigkeiten aus `requirements.txt`

### .env Beispiel

```env
CIVITAI_API_KEY=dein_api_key_hier
COMFYUI_MODEL_ROOT=/mnt/user/appdata/comfyui-nvidia/mnt/ComfyUI/models