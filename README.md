# Images → JSON Streamlit App

A lightweight, modular Streamlit app for turning images into structured JSON using an OpenAI‑compatible vision chat endpoint (no OCR).

## Features

- Clean 4‑page flow (Settings, Upload, Templates, Run & Test).
- Provider profiles with models.dev integration, capability probe, defaults, and secure key storage (session or Fernet‑encrypted).
- Template management with schema editor, few‑shots, import/export (YAML/JSON), and live preview rendering.
- Upload and manage images with per‑image tags (doc_type, locale) and selection.
- Run pipeline with robust fallbacks (tools → JSON mode → prompt) and strict JSON validation + auto‑repair.
- Cost estimation from provider usage and models.dev pricing.

## Quickstart

- Python 3.12 recommended. Create and activate a virtualenv.
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Configure environment (optional):

  ```bash
  cp .env.example .env
  # Edit .env with your API base/key/model
  # (Optional) generate APP_KMS_KEY for encrypted key storage
  ```

- (Optional) add `.streamlit/secrets.toml` for deployment secrets.

- Run the app:

  ```bash
  streamlit run app/main.py
  ```

On first run, `data/app.db` (SQLite) is created.

## App Flow

- Settings (⚙):
  - Manage provider profiles: base URL, API key, model ID, headers, timeout.
  - Security: store API key in session or encrypted in DB (Fernet via `APP_KMS_KEY`).
  - models.dev: lookup by model ID, show provider info and pricing, apply caps/pricing and logo.
  - Capability probe: tests JSON mode, vision, and tools; badges show detected capabilities.
  - Defaults: temperature, top_p, max output tokens, and preferences for JSON mode and tools.

- Upload (📤):
  - Upload multiple images (PNG/JPG), persisted to `data/uploads/`.
  - See thumbnails, select subset, and add per‑image tags (`doc_type`, `locale`).
  - Guard banner if active model lacks image modality.

- Templates (🧩):
  - Create/clone/delete templates; set system and user prompts.
  - Schema editor with validation; attach up to 3 few‑shot examples.
  - Import/export YAML/JSON; live preview renders final message array.

- Run & Test (🚀):
  - Choose a template and images, review the request plan from capabilities + defaults.
  - Run extraction per image using robust fallbacks; validate JSON with auto‑repair.
  - Show raw text, validated JSON, repair attempts, usage (if present), cost, and latency.
  - Save runs; define a test suite with assertions or golden JSON; run across profiles and export results.

## Pricing

- Usage‑based cost is computed from the provider’s `usage` object (prompt/completion tokens, optional cache read/write) and the active profile’s models.dev `cost` block. If usage is missing, cost shows as N/A.

## Limitations

- Vision‑only via Chat Completions; no OCR or PDF parsing is included.
- Server behaviors vary; tools/JSON mode may be unavailable. The app falls back automatically and surfaces readable errors.

## Structure

- `app/main.py` – App entry and sidebar chip.
- `app/pages/` – 4 pages (Settings, Upload, Templates, Run & Test).
- `app/core/` – Storage, provider gateway, templating, JSON enforcement, cost, UI.
- `app/assets/` – Static assets, including sample templates.
- `data/` and `export/` – Runtime dirs (gitignored).

## Sample Template & Images

- Sample template: `app/assets/templates/invoice_sample.yaml`.
- Bring your own sample images (we do not ship copyrighted materials). Upload in the Upload page and tag each as needed.
