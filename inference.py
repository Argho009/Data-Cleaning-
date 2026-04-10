from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
import requests


DEFAULT_SEEDS = {"task_easy": 42, "task_medium": 43, "task_hard": 44}
ENV_NAME = "data-cleaning-agent"

_SERVER_WAIT_ATTEMPTS = 30
_SERVER_WAIT_DELAY = 3.0


def _log_debug(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _clamp_score(score: float) -> float:
    return float(max(0.001, min(0.999, score)))


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
        _log_debug(f"[WARNING] LLM action failed: {e}. Using deterministic fallback.")
        return fallback


def _wait_for_server(env_base: str, timeout_s: float) -> bool:
    for attempt in range(_SERVER_WAIT_ATTEMPTS):
        try:
            r = requests.get(f"{env_base}/health", timeout=timeout_s)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        _log_debug(f"[WAIT] Server not ready, attempt {attempt + 1}/{_SERVER_WAIT_ATTEMPTS}...")
        time.sleep(_SERVER_WAIT_DELAY)
    return False


def run_task(
    env_base: str,
    llm_client: Any,
    model_name: str,
    task_id: str,
    seed: int,
    timeout_s: float,
) -> float:
    print(f"[START] task={task_id} env={ENV_NAME} model={model_name}", flush=True)

    step = 0
    rewards: List[float] = []
    final_score = 0.001
    success = False
    last_error: Optional[str] = None

    try:
        r = requests.post(
            f"{env_base}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=timeout_s,
        )
        r.raise_for_status()
        observation = r.json()
        done = False

        while not done:
            step += 1
            st = requests.get(f"{env_base}/state", timeout=timeout_s)
            st.raise_for_status()
            state_payload = st.json()

            action = _llm_choose_action(
                llm_client, model_name, task_id, step, observation, state_payload
            )

            resp = requests.post(f"{env_base}/step", json=action, timeout=timeout_s)
            resp.raise_for_status()
            payload = resp.json()

            reward_val = _clamp_score(float(payload["reward"]["score"]))
            done = bool(payload["done"])
            last_error = None
            observation = payload.get("observation", observation)
            rewards.append(reward_val)

            print(
                f"[STEP] step={step} action={action['action_type']} "
                f"reward={reward_val:.2f} done={'true' if done else 'false'} "
                f"error=null",
                flush=True,
            )

        final_score = _clamp_score(rewards[-1]) if rewards else 0.001
        success = True

    except Exception as e:
        last_error = str(e)
        _log_debug(f"[ERROR] Exception during task_id={task_id}: {e}")
        fallback_reward = 0.001
        rewards.append(fallback_reward)
        step = max(step, 1)
        print(
            f"[STEP] step={step} action=submit reward={fallback_reward:.2f} "
            f"done=true error={last_error}",
            flush=True,
        )
        final_score = 0.001
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={'true' if success else 'false'} steps={step} "
        f"score={final_score:.4f} rewards={rewards_str}",
        flush=True,
    )

    return final_score


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenAI-client baseline for Data Cleaning Agent OpenEnv."
    )
    parser.add_argument(
        "--env-base",
        default=os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860"),
        help="OpenEnv server base URL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override all task seeds.",
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
    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()
    )

    llm_client = None
    if api_base_url and model_name and api_key:
        try:
            llm_client = OpenAI(base_url=api_base_url, api_key=api_key)
        except Exception as e:
            _log_debug(f"[WARNING] Failed to initialize OpenAI client: {e}")
    else:
        _log_debug("[WARNING] Missing LLM env vars. Using fallback deterministic policy.")

    if not _wait_for_server(env_base, args.timeout):
        _log_debug("[ERROR] Environment server did not become ready in time.")

    scores: Dict[str, float] = {}
    for t in ["task_easy", "task_medium", "task_hard"]:
        seed = args.seed if args.seed is not None else DEFAULT_SEEDS[t]
        scores[t] = run_task(env_base, llm_client, model_name, t, seed, args.timeout)

    scores = {k: _clamp_score(v) for k, v in scores.items()}
    mean_score = _clamp_score(sum(scores.values()) / max(1, len(scores)))
    out = {"scores": scores, "mean": round(mean_score, 6)}
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())