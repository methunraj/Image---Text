# Repository Guidelines

## Project Structure & Modules
- `app/main.py`: Streamlit entry point and bootstrap.
- `app/pages/`: UI pages (ordered with numeric prefixes, e.g., `1_Upload_and_Process.py`, `2_Settings.py`).
- `app/core/`: Core logic (storage, provider gateway, templating, JSON validation, cost, UI helpers).
- `app/assets/`: Static assets and sample templates (`app/assets/templates/`).
- `data/`, `export/`: Runtime dirs (SQLite `data/app.db`, uploads, exports). Both are gitignored.

## Build, Run, and Dev Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Configure env: `cp .env.example .env` (set API base/key/model; optionally set `APP_KMS_KEY`).
- Run locally: `streamlit run app/main.py`
- Secrets (deploy): add `.streamlit/secrets.toml` (keys, `APP_PROFILE`).

## Coding Style & Naming
- Python 3.12, PEP 8, 4‑space indentation; add type hints where practical.
- Modules: `lower_snake_case.py`; functions/vars: `snake_case`; classes: `PascalCase`.
- Pages: prefix with order number, e.g., `3_Templates.py`, `4_Run_and_Test.py`.
- Keep UI code in `app/pages/`; put business logic in `app/core/` and import from pages.

## Testing Guidelines
- No repository test framework is configured yet. Prefer manual verification via the app (Run & Test flows).
- If adding tests, use pytest; name files `tests/test_*.py`; run with `pytest -q`.
- Keep tests fast and deterministic; mock network calls; avoid touching `data/app.db` (use temp files).

## Commit & PR Guidelines
- Use Conventional Commits style: `feat: add invoice schema editor`, `fix: handle missing usage cost`.
- PRs should include: clear description, linked issues, reproduction steps (`streamlit run app/main.py`), and screenshots for UI changes.
- Update docs when needed (README, sample templates). If you add env vars, update `.env.example` and mention them in README.
- When adding deps, update `requirements.txt` and justify the choice.

## Security & Configuration
- Never commit secrets; prefer `.env` locally and `.streamlit/secrets.toml` in deployment.
- API keys can be stored in session or encrypted in DB (set `APP_KMS_KEY` to enable Fernet encryption).
- Avoid committing large or sensitive images; uploads live under `data/uploads/` at runtime.

