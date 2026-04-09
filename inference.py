"""
OpenAI-client baseline policy for reproducible benchmark scores.

Runs the HTTP OpenEnv API: POST /reset, POST /step, GET /state.
LLM calls are made through the OpenAI client with API_BASE_URL/MODEL_NAME.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from openai import OpenAI
import requests


DEFAULT_SEEDS = {"task_easy": 42, "task_medium": 43, "task_hard": 44}


def _choose_action(task_id: str, step_count: int) -> Dict[str, Any]:
    policies: Dict[str, List[Dict[str, Any]]] = {
        "task_easy": [
            {"action_type": "fix_dates", "parameters": {}},
            {"action_type": "remove_duplicates", "parameters": {}},
            {"action_type": "submit", "parameters": {}},
        ],
        "task_medium": [
            {
                "action_type": "fill_nulls",
                "parameters": {"strategy": "mode", "columns": ["city", "zip_code"]},
            },
            {"action_type": "standardize", "parameters": {"columns": ["state"]}},
            {"action_type": "submit", "parameters": {}},
        ],
        "task_hard": [
            {"action_type": "fix_numeric", "parameters": {"columns": ["salary"]}},
            {"action_type": "standardize", "parameters": {"columns": ["phone"]}},
            {"action_type": "detect_fuzzy_duplicates", "parameters": {}},
            {"action_type": "remove_outliers", "parameters": {}},
            {"action_type": "submit", "parameters": {}},
        ],
    }
    script = policies[task_id]
    idx = min(step_count - 1, len(script) - 1)
    return script[idx]


def _llm_choose_action(
    client: Any,
    model_name: str,
    task_id: str,
    step_count: int,
    observation: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    # Keep a deterministic fallback to preserve reproducibility.
    fallback = _choose_action(task_id, step_count)
    if client is None:
        return fallback

    prompt = {
        "task_id": task_id,
        "step_count": step_count,
        "available_actions": observation.get("available_actions", []),
        "errors_remaining": observation.get("errors_remaining"),
        "error_report": observation.get("error_report", []),
        "max_steps": observation.get("max_steps"),
        "progress_ratio": state.get("progress_ratio"),
        "instruction": (
            "Return one JSON object with keys action_type and parameters. "
            "No markdown, no extra keys."
        ),
    }
    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a precise data-cleaning agent."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            timeout=10.0,
        )
        content = completion.choices[0].message.content or "{}"
        parsed = json.loads(content)
        action_type = parsed.get("action_type")
        params = parsed.get("parameters", {})
        if not isinstance(action_type, str):
            return fallback
        if not isinstance(params, dict):
            params = {}
        return {"action_type": action_type, "parameters": params}
    except Exception as e:
        print(f"[WARNING] LLM action failed: {e}. Using deterministic fallback.")
        return fallback


def run_task(
    env_base: str,
    llm_client: Any,
    model_name: str,
    task_id: str,
    seed: int,
    timeout_s: float,
) -> float:
    print(f"[START] task_id={task_id} seed={seed}")
    import time
    try:
        r = None
        for attempt in range(5):
            try:
                r = requests.post(
                    f"{env_base}/reset",
                    json={"task_id": task_id, "seed": seed},
                    timeout=timeout_s,
                )
                r.raise_for_status()
                break
            except requests.exceptions.ConnectionError as e:
                if attempt == 4:
                    raise e
                print(f"[RETRY] Waiting for environment server... attempt {attempt + 1}")
                time.sleep(2)
                
        if r is None:
            return 0.0

        observation = r.json()
        done = False
        step = 0
        final_score = 0.0
        while not done:
            step += 1
            st = requests.get(f"{env_base}/state", timeout=timeout_s)
            st.raise_for_status()
            state_payload = st.json()
            action = _llm_choose_action(llm_client, model_name, task_id, step, observation, state_payload)
            resp = requests.post(f"{env_base}/step", json=action, timeout=timeout_s)
            resp.raise_for_status()
            payload = resp.json()
            reward = payload["reward"]
            done = bool(payload["done"])
            final_score = float(reward["score"])
            observation = payload.get("observation", observation)
            st = requests.get(f"{env_base}/state", timeout=timeout_s)
            st.raise_for_status()
            prog = st.json().get("progress_ratio", None)
            print(
                f"[STEP] step={step} action={action['action_type']} "
                f"score={final_score:.4f} progress={prog} done={str(done).lower()}"
            )
        print(f"[END] task_id={task_id} final_score={final_score:.4f}")
        return final_score
    except Exception as e:
        print(f"[ERROR] Exception during task_id={task_id}: {e}")
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI-client baseline for Data Cleaning Agent OpenEnv.")
    parser.add_argument(
        "--env-base",
        default=os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860"),
        help="OpenEnv server base URL (env ENV_BASE_URL overrides default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override all task seeds (default: built-in per-task seeds for reproducibility).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout seconds per request.",
    )
    args = parser.parse_args()
    env_base = args.env_base.rstrip("/")

    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "fallback-model").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()
    
    llm_client = None
    if api_base_url and model_name and api_key:
        try:
            llm_client = OpenAI(base_url=api_base_url, api_key=api_key)
        except Exception as e:
            print(f"[WARNING] Failed to initialize OpenAI client: {e}")
    else:
        print("[WARNING] Missing LLM env vars. Using fallback deterministic policy.")

    scores: Dict[str, float] = {}
    for t in ["task_easy", "task_medium", "task_hard"]:
        seed = args.seed if args.seed is not None else DEFAULT_SEEDS[t]
        scores[t] = run_task(env_base, llm_client, model_name, t, seed, args.timeout)

    out = {"scores": scores, "mean": round(sum(scores.values()) / max(1, len(scores)), 6)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
