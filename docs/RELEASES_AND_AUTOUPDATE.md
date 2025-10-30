# Releases & Auto-Updates (Tauri + GitHub Releases)

This app ships as a **native desktop** (Tauri) bundle that wraps the Streamlit server.  
Auto-updates are handled by **tauri-plugin-updater** using a simple JSON feed (`latest.json`)
published on **GitHub Releases** by CI.

---

## 0) What ships where

- **App binaries** (DMG for macOS, EXE/MSI for Windows) are uploaded to the **GitHub Release** for each tag.
- CI also uploads a signed **`latest.json`** pointing to those binaries.
- The desktop app reads that feed on startup (and when the user opens Admin -> Updates box).

---

## 1) One-time setup

1. **Generate updater keys (locally)**
   ```bash
   cargo install tauri-cli --locked
   tauri signer generate
   ```

   Copy the public key into `src-tauri/tauri.conf.json` -> `plugins.updater.pubkey`.

   Add the private key and password to GitHub Secrets:

   - `TAURI_PRIVATE_KEY`
   - `TAURI_KEY_PASSWORD`

2. **Set the updater feed URL**

   In `src-tauri/tauri.conf.json` -> `plugins.updater.endpoints[0]`:

   ```
   https://github.com/<OWNER>/<REPO>/releases/latest/download/latest.json
   ```

   In `src-tauri/src/main.rs` (File 18 patch), keep `UPDATER_FEED_URL` env in `spawn_streamlit_sidecar(...)`
   in sync with the same URL.

3. **CI workflow**

   `/.github/workflows/tauri-release.yml` (File 10) is already configured to:

   - build Win/macOS bundles,
   - upload artifacts + `.sig`,
   - generate and upload `latest.json`.

---

## 2) Cutting a release

1. Bump version in `src-tauri/tauri.conf.json` and (optionally) your Python app banner.
2. Commit & tag:

   ```bash
   git add -A
   git commit -m "chore(release): v0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```

3. CI runs -> creates a Release, uploads installers & signatures, and publishes `latest.json`.
4. Users will see "Update available" on next launch and can install in-app.

---

## 3) Verifying the update

1. Open the app -> Admin Console -> Updates expander.
2. Shows Current version vs Latest available.
3. Provides a Download installer button for your platform.
4. Alternatively, open the feed directly:

   ```
   https://github.com/<OWNER>/<REPO>/releases/latest/download/latest.json
   ```

---

## 4) Local testing & troubleshooting

### A) Test without tagging (advanced)

Publish a draft Release manually with assets and `.sig`, upload a hand-crafted `latest.json`.

Temporarily point the app to a test feed by setting env before starting Tauri:

```bash
UPDATER_FEED_URL=https://your-temp-host/latest.json
```

(You can also hardcode a test URL in `main.rs` while testing, then revert.)

### B) Common pitfalls

- Updater says "no update"  
  Ensure version in `latest.json` is greater than the app's `APP_VERSION`.  
  The app's version comes from `tauri.conf.json` version.

- Signature mismatch  
  Make sure CI uploads the `.sig` produced by Tauri for each asset and `latest.json`
  uses the exact matching signature strings. Do not reuse signatures across files.

- Feed not reachable / 404  
  The release must be published (not draft) and `latest.json` must be attached.  
  Double-check the endpoint URL in both `tauri.conf.json` and `main.rs`.

- macOS architectures  
  If you build arm64 only, `latest.json` should include `darwin-aarch64`.  
  If universal builds are not set, provide separate x64/arm64 DMGs and include both.

---

## 5) Security notes

- Keep `TAURI_PRIVATE_KEY` only in GitHub Secrets (never commit it).
- `latest.json` is signed; the app verifies signatures before installing.
- Supabase service keys remain in Supabase; desktop app never bundles them.
- Provider API keys:
  - Project-scoped keys are client-encrypted (Fernet) and stored only as ciphertext in Supabase.
  - Provider-wide keys (if used) are encrypted server-side (AES-GCM) via Edge Functions.

---

## 6) Rotating updater keys

1. Generate a new keypair with `tauri signer generate`.
2. Update `pubkey` in `tauri.conf.json` and redeploy the app in a new version.
3. Update GitHub Secrets with the new private key and password.
4. Tag a new release; old apps will continue to verify existing assets using the old key.  
   For smooth rotation, plan a short transition where both old and new builds are available.

---

## 7) Release cadence & versioning

- Use SemVer (e.g., `v0.2.1`) and keep changelogs in your Release notes.
- For risky changes (new models, auth tweaks), cut a beta tag (`v0.3.0-beta.1`) and test internally.

---

## 8) Edge Functions recap (admin-only)

Deployed via:

```bash
supabase secrets set SUPABASE_SERVICE_ROLE_KEY=<service_role_key>
supabase secrets set PROVIDER_SECRET_PASSPHRASE='<long_random_phrase>'
supabase functions deploy admin_add_user
supabase functions deploy save_provider_secret
supabase functions deploy test_model_connection
# optional
supabase functions deploy import_config_xlsx
```

- `admin_add_user` -- invites/creates a user + upserts `user_profiles`.
- `save_provider_secret` -- encrypts and stores provider API key (AES-GCM).
- `test_model_connection` -- decrypts key, pings provider/model.
- `import_config_xlsx` (optional) -- migrates legacy `config.xlsx` -> Supabase.

---

## 9) Environment summary

Desktop app expects:

```bash
SUPABASE_URL=https://<project-id>.supabase.co
SUPABASE_ANON_KEY=<anon>
APP_DB_PATH=./data/app.db
APP_KMS_KEY=<fernet-key>  # or APP_KMS_KEY_FILE=./data/kms.key
SUPABASE_PERSIST_SESSION=1
```

Tauri injects at runtime (File 18 patch):

```bash
APP_VERSION=<from tauri.conf.json>
UPDATER_FEED_URL=https://github.com/<OWNER>/<REPO>/releases/latest/download/latest.json
```

---

## 10) Release checklist (DoD)

- Bumped version in `tauri.conf.json`
- CI secrets present (`TAURI_PRIVATE_KEY`, `TAURI_KEY_PASSWORD`)
- `pubkey` in `tauri.conf.json` matches the private key in GH Secrets
- Tag pushed (`vX.Y.Z`) -> Release created with assets
- `latest.json` attached by CI and valid for all target platforms
- Manual smoke test: install -> app starts -> Updates box shows current vs latest

---

That is the release/update playbook.  
Want me to generate a `.env.example` (File 20) and a quickstart script that sets sane dev defaults for your team?
