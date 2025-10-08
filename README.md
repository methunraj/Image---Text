# Images → JSON Streamlit App

A lightweight, modular Streamlit app for turning images into structured JSON using an OpenAI‑compatible vision chat endpoint (no OCR).

## Features

- Clean 2‑page flow (Settings, Upload & Process).
- Server-side model registry maintained in `config/models.xlsx` (profiles, capabilities, per-million pricing, limits).
- Optional encrypted secrets managed via `scripts/secretset.py` + Fernet/`APP_KMS_KEY`.
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
  # Edit .env to set APP_PROFILE (dev/prod) and optional overrides
  # (Optional) generate APP_KMS_KEY or let scripts/secretset.py create data/kms.key
  ```

- Populate provider API keys referenced in `config/models.xlsx` (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `DASHSCOPE_API_KEY`). Providers without credentials are skipped automatically; the registry falls back to the first available provider/model.

- Edit `config/models.xlsx` to manage providers/models (per-million pricing, capabilities, reasoning defaults). Use `APP_PROFILE` to switch profiles. Set `compat_max_tokens_param` when a provider expects a non-standard max-token field (e.g., `maxOutputTokens` for Gemini), and toggle `compat_allow_input_image` to `N` for providers that reject the OpenAI `input_image` fallback.
- (Optional) store encrypted secrets: `python scripts/secretset.py OPENAI_API_KEY sk-...`

- Run the app:

  ```bash
  streamlit run app/main.py
  ```

On first run, `data/app.db` (SQLite) is created.

## App Flow

- Settings (⚙):
  - Configure providers/models (base URL, headers, timeouts) and manage API keys (session or encrypted).
  - Browse/select models from the catalog; probe capabilities; set an active profile in the local DB.

- Upload & Process (📤):
  - Upload or select folders of images, tag images, select a template, and run extraction.
  - View raw or structured output; export JSON/Markdown/XLSX/DOCX; usage and cost if available.
  - Upload PDF(s): convert pages to images into a chosen folder (DPI/format/page range), update a folder checkpoint, and queue pending pages for processing.

## Pricing

- Usage‑based cost is computed from the provider `usage` payload (prompt/completion/cache tokens) using the active model’s per-1K prices (auto-converted from the Excel per-million rates). If usage is missing, cost shows as N/A.

## Limitations

- Vision‑only via Chat Completions; no OCR or PDF parsing is included.
- Server behaviors vary; tools/JSON mode may be unavailable. The app falls back automatically and surfaces readable errors.

## Structure

- `app/main.py` – App entry and sidebar chip.
- `app/pages/` – 2 pages (Settings, Upload & Process).
- `app/core/` – Storage, provider gateway, templating, JSON enforcement, cost, UI.
- `config/models.xlsx` – Excel workbook defining profiles, providers, models, pricing, reasoning defaults.
- `app/assets/` – Static assets, including sample templates. YAML files under `app/assets/templates/` are auto-imported on startup and updated when templates are saved in the Settings page.
- `data/` and `export/` – Runtime dirs (gitignored).

## Sample Template & Images

- Sample template: `app/assets/templates/invoice_sample.yaml`.
- Bring your own sample images (we do not ship copyrighted materials). Upload in the Upload page and tag each as needed.
