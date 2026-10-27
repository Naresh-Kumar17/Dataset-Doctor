# Dataset Doctor: Authentication & Security Architecture

## 1. Security Philosophy: Local-First & Zero-Trust Ingestion

Dataset Doctor is designed primarily as a local-first, developer-centric tool. Ingested datasets often contain proprietary, confidential, or sensitive personal data. Our architecture is governed by three strict security principles:
1. **Zero External Telemetry**: Uploaded datasets are never transmitted to external services, cloud storage, or third-party APIs. All profiling, model evaluation, and transformations execute locally within the user's Python runtime.
2. **Defensive File Ingestion**: Untrusted CSV files are treated as potentially malicious payloads. Ingestion enforces strict path traversal mitigation, stream chunk size bounds, MIME verification, and CSV formula injection neutralization.
3. **Ephemeral Session Lifecycle**: Datasets, intermediate partitions, and models are bound to volatile session tokens and purged upon session termination or server shutdown.

---

## 2. Ingestion Defense & Upload Hardening

### 2.1. File Size & Stream Metering
- **Hard Limit**: 50 MB maximum file size (`MAX_FILE_SIZE_MB = 50`).
- **Streaming Guard**: Uploads are read in 1 MB chunks. If the cumulative byte count exceeds 50 MB, the stream is aborted immediately with HTTP `413 Payload Too Large` before memory exhaustion can occur.

### 2.2. Path Traversal & Filename Sanitization
- Filenames extracted from multipart headers are strictly passed through `werkzeug.utils.secure_filename` and stripped of directory traversal tokens (`../`, `..\`, absolute prefixes, null bytes).
- Uploaded files are written to isolated session directories inside the temporary OS workspace:
  ```python
  safe_name = secure_filename(upload_file.filename)
  session_path = os.path.join(temp_dir, session_id, safe_name)
  ```

### 2.3. CSV Formula Injection (DDE) Mitigation
- Spreadsheet software (Microsoft Excel, LibreOffice Calc) automatically interprets cells beginning with `=`, `+`, `-`, or `@` as executable formulas.
- In `doctor/report.py` and exported cleaned CSV files, cells with leading formula triggers are escaped or formatted safely to prevent Dynamic Data Exchange (DDE) exploitation.

---

## 3. Session Security & Access Control

### 3.1. Session Token Architecture (`app/session.py`)
- Sessions are keyed by cryptographically secure UUID4 identifiers:
  ```python
  session_id = secrets.token_urlsafe(32)
  ```
- Stored as HTTP-only, SameSite=Strict cookies to protect against Cross-Site Scripting (XSS) and unauthorized script extraction.

### 3.2. Authentication Layer (`app/auth.py` & `app/security.py`)
- **Local Development Mode**: Single-tenant, frictionless session access without passwords.
- **Hosted / Multi-User Mode**: Optional HTTP Basic Auth or Bearer Token authentication via environment variable configuration:
  ```env
  DATASET_DOCTOR_AUTH_ENABLED=true
  DATASET_DOCTOR_ADMIN_USER=admin
  DATASET_DOCTOR_ADMIN_PASSWORD_HASH=<bcrypt_hash>
  ```
- **Rate Limiting**: Critical endpoints (`/upload`, `/decide/*/apply`) enforce in-memory rate limiting (max 30 requests per minute per IP) to guard against Denial-of-Service (DoS).

---

## 4. Privacy & Sensitive Data Safeguards

1. **No Log Exfiltration**: Logs emitted to `docs/gate_log.md` and standard output record aggregate statistics, error types, and column names—never raw row records, values, or personally identifiable information (PII).
2. **Deterministic Seed Disclosure**: Random states are fixed and explicitly recorded for evaluation integrity without exposing data content.
3. **User Confirmation for Destructive Actions**: Non-reversible actions (e.g. dropping columns or dropping rows) require explicit user approval and emit before/after count verifications.
