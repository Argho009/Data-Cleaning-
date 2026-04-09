---
title: Data Cleaning Agent
emoji: "🧹"
colorFrom: indigo
colorTo: blue
sdk: docker
sdk_version: "latest"
python_version: "3.12"
app_file: app.py
pinned: false
tags:
  - openenv
---

# Data Cleaning Agent (OpenEnv)

`Data Cleaning Agent` is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-style environment for **real-world tabular data quality**: an agent applies discrete cleaning operations to dirty synthetic datasets (sales, addresses, HR records) and receives dense, interpretable rewards. The server exposes the standard control plane: **`reset` → `step` → `state`**.

## Why this environment

Data teams spend much of their time on schema drift, nulls, inconsistent formats, duplicates, and outliers. This project turns that workflow into a **reproducible benchmark** with procedural data, typed actions, automated graders per scenario, and partial credit during episodes.

## Project layout

```text
├── environment.py   # DataCleaningEnv: Observation, Action, Reward, EnvState; step/reset/state
├── tasks.py         # Three scenarios + detect_errors() + grade_task() metrics
├── app.py           # FastAPI: POST /reset, POST /step, GET /state, GET /health (+ optional /ui/*)
├── openenv.yaml     # OpenEnv metadata: spaces, metrics, API map
├── inference.py     # OpenAI-client baseline with reproducible task scores
├── Dockerfile       # Hugging Face Spaces / Docker (port 7860)
├── requirements.txt
└── README.md
```

## Core API (OpenEnv control plane)

| Method | Path | Body / notes |
|--------|------|----------------|
| `POST` | `/reset` | `{"task_id": "task_easy" \| "task_medium" \| "task_hard", "seed": optional int}` |
| `POST` | `/step` | `{"action_type": "...", "parameters": { ... }}` |
| `GET` | `/state` | Typed `EnvState` (episode progress, errors, seed) |
| `GET` | `/health` | `{"status":"ok"}` |

`seed` fixes procedural data for reproducible evaluation; default seeds per task are `42`, `43`, `44` when omitted.

## Observation space (`Observation`)

| Field | Meaning |
|-------|---------|
| `task_id` | Scenario id |
| `description` | Natural-language task brief |
| `current_data` | JSON array of row objects (pandas `orient="records"`) |
| `target_schema` | Column → expected dtype or format hint |
| `error_report` | Human-readable issue strings |
| `errors_remaining` / `errors_initial` | Counts for progress signals |
| `available_actions` | Allowed `action_type` values |
| `step_count` / `max_steps` | Horizon |

## Action space (`Action`)

- `action_type`: one of `fix_dates`, `remove_duplicates`, `fill_nulls`, `standardize`, `fix_numeric`, `detect_fuzzy_duplicates`, `remove_outliers`, `inspect`, `submit`.
- `parameters`: optional dict (e.g. `columns`, `strategy`, `subset`, `mapping`).

## State space (`EnvState`)

Returned by `GET /state`. Includes `ready`, `task_id`, `seed`, `step_count`, `max_steps`, `done`, `rows`, `errors`, `errors_remaining`, `errors_initial`, and `progress_ratio` (0–1, error-mass cleared vs episode start).

## Reward (0.0–1.0)

- **During an episode** (before terminal): reward blends **fraction of initial issues resolved**, **remaining-error ratio**, **action hints** (sensible ops for the scenario), and **step budget pressure** (shorter episodes are slightly favored). See `Reward.breakdown` for `fraction_errors_fixed`, `progress_ratio`, `step_budget_ratio`, `step_delta`.
- **On submit or max steps**: terminal score combines automated **task-grade** metrics (see `openenv.yaml`), **error-fix ratio**, **schema compliance**, **valid-row preservation**, and **step efficiency**.

## Tasks and graders (easy → hard)

| Task | Focus | Key grader outputs (0–1) |
|------|--------|---------------------------|
| `task_easy` | Mixed `sale_date` formats + exact duplicate sales | `date_parse_accuracy`, `duplicate_quality` |
| `task_medium` | Null city/zip + inconsistent US states | `null_fill_rate`, `standardization_accuracy`, `schema_compliance` |
| `task_hard` | Fuzzy name dupes, corrupted salary strings, phone format, age outliers | `fuzzy_dedup_f1`, `numeric_fix_accuracy`, `outlier_removal_quality`, `zero_false_deletion_score` |

Date parsing uses mixed-format `pandas` parsing (`format="mixed"`) so ISO and locale-style literals are handled consistently.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 7860
```

Or: `python "python run_server.py"` (starts the same app on `127.0.0.1:7860`).

## Baseline inference (`inference.py`)

The baseline is the required root-level `inference.py`. It:

- uses the OpenAI client for all LLM calls,
- reads required variables from environment configuration,
- emits structured logs in `[START]`, `[STEP]`, `[END]` format,
- runs all three tasks with fixed default seeds for reproducibility.

Required environment variables:

- `API_BASE_URL`: OpenAI-compatible LLM endpoint (provider gateway URL).
- `MODEL_NAME`: model identifier used by the OpenAI client.
- `HF_TOKEN`: Hugging Face/API key (fallback if `OPENAI_API_KEY` is unset).
- `OPENAI_API_KEY`: OpenAI-compatible API key (preferred when set).

Optional:

- `ENV_BASE_URL`: environment server URL (default `http://127.0.0.1:7860`).

Run:

```powershell
python inference.py --env-base http://127.0.0.1:7860
```

Optional: `--seed 123` overrides every task’s seed; `--timeout` sets HTTP timeouts.

Example output (your exact numbers may vary slightly with library versions; seeds fix the data):

```json
{
  "scores": {
    "task_easy": 1.0,
    "task_medium": 1.0,
    "task_hard": 0.8959
  },
  "mean": 0.9653
}
```

## Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

Health: `GET http://localhost:7860/health` (image includes a healthcheck).

## Hugging Face Spaces

1. Create a **Docker** Space.
2. Push this repo (or upload files): `Dockerfile`, `requirements.txt`, `app.py`, `environment.py`, `tasks.py`, `openenv.yaml`, `README.md`, etc.
3. Add the Space tag `openenv`.
4. **App port**: `7860` (matches `EXPOSE` and `CMD`).
5. No GPU required. For remote baseline runs, set:
   - `ENV_BASE_URL` to your deployed Space URL,
   - `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`/`OPENAI_API_KEY` for inference.

## Pre-submission checklist (required)

- `openenv validate` passes.
- `docker build` and `docker run` start cleanly.
- `POST /reset`, `POST /step`, and `GET /state` respond correctly.
- `inference.py` completes under 20 minutes on CPU and prints valid `[START]/[STEP]/[END]` logs.
- All three tasks return grader-backed scores in `[0.0, 1.0]`.

## Notes

- The FastAPI app also serves optional **`/ui/*`** routes for interactive CSV cleaning; RL training should use **`/reset` / `/step` / `/state`** only.
- `openenv.yaml` documents the full metadata surface for OpenEnv tooling and Hub cards.
