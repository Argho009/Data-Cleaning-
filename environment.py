from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from tasks import TaskBundle, build_task, detect_errors, grade_task


def _cb(v: float) -> float:
    """Clamp to strictly open interval (0, 1) as required by OpenEnv validator."""
    return float(max(0.001, min(0.999, v)))


AVAILABLE_ACTIONS = [
    "fix_dates",
    "remove_duplicates",
    "fill_nulls",
    "standardize",
    "fix_numeric",
    "detect_fuzzy_duplicates",
    "remove_outliers",
    "inspect",
    "submit",
]


class Observation(BaseModel):
    task_id: str
    description: str
    current_data: str
    target_schema: Dict[str, str]
    error_report: List[str]
    errors_remaining: int
    errors_initial: int
    available_actions: List[str]
    step_count: int
    max_steps: int


class Action(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    score: float
    breakdown: Dict[str, float]
    message: str
    done: bool

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Enforce strict (0, 1) bounds required by OpenEnv validator."""
        return float(max(0.001, min(0.999, v)))


class EnvState(BaseModel):
    """Typed episode state for OpenEnv-compatible clients."""

    ready: bool
    task_id: Optional[str] = None
    seed: Optional[int] = None
    step_count: int = 0
    max_steps: int = 15
    done: bool = False
    rows: int = 0
    errors: List[str] = Field(default_factory=list)
    errors_remaining: int = 0
    errors_initial: int = 0
    progress_ratio: float = 0.0


class DataCleaningEnv:
    def __init__(self, max_steps: int = 15) -> None:
        self.max_steps = max_steps
        self.task: Optional[TaskBundle] = None
        self.current_df: Optional[pd.DataFrame] = None
        self.step_count = 0
        self.done = False
        self.total_errors_at_start = 1
        self._episode_seed: Optional[int] = None

    def _observation(self) -> Observation:
        if self.task is None or self.current_df is None:
            raise RuntimeError("Environment not reset")
        errs = detect_errors(self.task.task_id, self.current_df)
        return Observation(
            task_id=self.task.task_id,
            description=self.task.description,
            current_data=self.current_df.to_json(orient="records"),
            target_schema=self.task.target_schema,
            error_report=errs,
            errors_remaining=len(errs),
            errors_initial=self.total_errors_at_start,
            available_actions=AVAILABLE_ACTIONS,
            step_count=self.step_count,
            max_steps=self.max_steps,
        )

    def reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        self.task = build_task(task_id, seed=seed)
        self._episode_seed = seed
        self.current_df = self.task.df.copy(deep=True)
        self.step_count = 0
        self.done = False
        self.total_errors_at_start = max(1, len(detect_errors(task_id, self.current_df)))
        return self._observation()

    def _apply_action(self, action: Action) -> Tuple[bool, int, str]:
        assert self.current_df is not None
        assert self.task is not None
        before = self.current_df.copy(deep=True)
        message = ""
        at = action.action_type
        params = action.parameters or {}

        if at == "fix_dates" and "sale_date" in self.current_df.columns:
            # format="mixed" parses ISO and locale-style dates without breaking YYYY-MM-DD.
            self.current_df["sale_date"] = pd.to_datetime(
                self.current_df["sale_date"], errors="coerce", format="mixed"
            ).dt.strftime("%Y-%m-%d")
            message = "Date format normalized to YYYY-MM-DD."

        elif at == "remove_duplicates":
            subset = params.get("subset")
            if subset and isinstance(subset, list):
                self.current_df = self.current_df.drop_duplicates(subset=subset).reset_index(drop=True)
            elif self.task.task_id == "task_easy":
                self.current_df = self.current_df.drop_duplicates(
                    subset=["sale_id", "sale_date", "amount", "product"]
                ).reset_index(drop=True)
            elif self.task.task_id == "task_hard":
                self.current_df = self.current_df.drop_duplicates(
                    subset=["employee_name", "dob"]
                ).reset_index(drop=True)
            else:
                self.current_df = self.current_df.drop_duplicates().reset_index(drop=True)
            message = "Duplicate rows removed."

        elif at == "fill_nulls":
            strategy = params.get("strategy", "mode")
            columns = params.get("columns", [])
            if not columns:
                columns = [c for c in self.current_df.columns if self.current_df[c].isna().any()]
            for col in columns:
                if col not in self.current_df.columns:
                    continue
                if strategy == "mode":
                    mode = self.current_df[col].dropna().mode()
                    value = mode.iloc[0] if not mode.empty else "UNKNOWN"
                elif strategy == "constant":
                    value = params.get("value", "UNKNOWN")
                else:
                    value = "UNKNOWN"
                self.current_df[col] = self.current_df[col].fillna(value)
            message = "Null values filled."

        elif at == "standardize":
            mapping = params.get("mapping", {})
            columns = params.get("columns", [])
            if not columns:
                if "state" in self.current_df.columns:
                    columns = ["state"]
                if "phone" in self.current_df.columns:
                    columns.append("phone")

            for col in columns:
                if col not in self.current_df.columns:
                    continue
                if col == "state":
                    default_state_map = {
                        "CA": "CA",
                        "California": "CA",
                        "Calif.": "CA",
                        "ca": "CA",
                        "TX": "TX",
                        "Texas": "TX",
                        "Tex.": "TX",
                        "tx": "TX",
                    }
                    use_map = mapping.get("state", default_state_map)
                    self.current_df["state"] = self.current_df["state"].map(lambda x: use_map.get(x, x))
                elif col == "phone":
                    def _fmt_phone(p: Any) -> str:
                        digits = re.sub(r"\D", "", str(p))
                        if len(digits) >= 10:
                            digits = digits[-10:]
                            return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
                        return str(p)
                    self.current_df["phone"] = self.current_df["phone"].map(_fmt_phone)
            message = "Columns standardized."

        elif at == "fix_numeric":
            columns = params.get("columns", ["salary"])
            for col in columns:
                if col not in self.current_df.columns:
                    continue
                cleaned = (
                    self.current_df[col]
                    .astype(str)
                    .str.replace("O", "0", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                )
                self.current_df[col] = pd.to_numeric(cleaned, errors="coerce")
            message = "Numeric corruption fixed."

        elif at == "detect_fuzzy_duplicates" and self.task.task_id == "task_hard":
            self.current_df["name_key"] = self.current_df["employee_name"].astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
            self.current_df = self.current_df.sort_values(by=["dob", "name_key"]).drop_duplicates(
                subset=["dob", "name_key"], keep="first"
            ).drop(columns=["name_key"]).reset_index(drop=True)
            message = "Fuzzy duplicate candidates consolidated."

        elif at == "remove_outliers":
            if "age" in self.current_df.columns:
                self.current_df = self.current_df[(self.current_df["age"] >= 18) & (self.current_df["age"] <= 90)].reset_index(drop=True)
                message = "Outlier ages removed."

        elif at == "inspect":
            message = "Inspection action performed; no data mutation."

        elif at == "submit":
            message = "Submission requested."
        else:
            message = f"Action {at} had no effect for this task."

        changed = not before.equals(self.current_df)
        valid_deleted = 0
        if self.task and "row_id" in before.columns and "row_id" in self.current_df.columns:
            before_valid = set(before["row_id"]).intersection(self.task.valid_row_ids)
            after_valid = set(self.current_df["row_id"]).intersection(self.task.valid_row_ids)
            valid_deleted = len(before_valid - after_valid)
        return changed, valid_deleted, message

    def _final_reward(self) -> Tuple[float, Dict[str, float]]:
        assert self.task is not None and self.current_df is not None
        metrics = grade_task(self.task.task_id, self.current_df, self.task)
        remaining = len(detect_errors(self.task.task_id, self.current_df))
        errors_fixed = max(0, self.total_errors_at_start - remaining)
        errors_score = errors_fixed / max(1, self.total_errors_at_start)

        if self.task.task_id == "task_easy":
            task_grade = 0.5 * metrics["date_parse_accuracy"] + 0.5 * metrics["duplicate_quality"]
        elif self.task.task_id == "task_medium":
            task_grade = (
                0.3 * metrics["null_fill_rate"] +
                0.4 * metrics["standardization_accuracy"] +
                0.3 * metrics["schema_compliance"]
            )
        else:
            task_grade = (
                0.35 * metrics["fuzzy_dedup_f1"] +
                0.30 * metrics["numeric_fix_accuracy"] +
                0.20 * metrics["outlier_removal_quality"] +
                0.15 * metrics["zero_false_deletion_score"]
            )

        valid_rows_preserved = 0.001
        if "row_id" in self.current_df.columns:
            curr = set(self.current_df["row_id"]).intersection(self.task.valid_row_ids)
            valid_rows_preserved = _cb(len(curr) / max(1, len(self.task.valid_row_ids)))
        schema_compliance = _cb(metrics.get("schema_compliance", min(0.999, task_grade)))
        raw_bonus = 0.999 if self.step_count < (self.max_steps / 2) else max(0.001, 1.0 - (self.step_count / self.max_steps))
        steps_bonus = _cb(raw_bonus)
        errors_score_c = _cb(errors_score)

        final = _cb(
            0.40 * errors_score_c +
            0.30 * schema_compliance +
            0.20 * valid_rows_preserved +
            0.10 * steps_bonus
        )
        breakdown = {
            "task_grade": round(_cb(task_grade), 4),
            "errors_fixed_ratio": round(errors_score_c, 4),
            "schema_compliance_score": round(schema_compliance, 4),
            "valid_rows_preserved": round(valid_rows_preserved, 4),
            "steps_efficiency_bonus": round(steps_bonus, 4),
            "final_score": round(final, 4),
        }
        return final, breakdown

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if action.action_type not in AVAILABLE_ACTIONS:
            raise ValueError(f"Unsupported action_type: {action.action_type}")
        assert self.current_df is not None

        self.step_count += 1
        changed, valid_deleted, msg = self._apply_action(action)
        step_hint = 0.0
        breakdown: Dict[str, float] = {"step_delta": 0.001}

        correct_by_task = {
            "task_easy": {"fix_dates", "remove_duplicates", "inspect", "submit"},
            "task_medium": {"fill_nulls", "standardize", "inspect", "submit"},
            "task_hard": {"detect_fuzzy_duplicates", "fix_numeric", "standardize", "remove_outliers", "inspect", "submit"},
        }
        if action.action_type in correct_by_task.get(self.task.task_id if self.task else "", set()):
            step_hint += 0.06
        if valid_deleted > 0:
            step_hint -= 0.06 * valid_deleted
        if not changed and action.action_type not in {"inspect", "submit"}:
            step_hint -= 0.03

        assert self.task is not None
        remaining_errs = len(detect_errors(self.task.task_id, self.current_df))
        progress_ratio = _cb(1.0 - (remaining_errs / max(1, self.total_errors_at_start)))
        fraction_errors_fixed = _cb(
            (self.total_errors_at_start - remaining_errs) / max(1, self.total_errors_at_start)
        )
        step_budget_ratio = _cb(1.0 - (self.step_count / max(1, self.max_steps)))

        done = action.action_type == "submit" or self.step_count >= self.max_steps
        if done:
            self.done = True
            final_score, final_breakdown = self._final_reward()
            score = final_score
            breakdown.update(final_breakdown)
            breakdown["progress_ratio"] = round(progress_ratio, 4)
            message = "Episode finished."
        else:
            intermediate = (
                0.52 * fraction_errors_fixed
                + 0.28 * progress_ratio
                + 0.12 * max(0.0, step_hint)
                + 0.08 * step_budget_ratio
            )
            score = float(max(0.001, min(0.999, intermediate)))
            breakdown.update(
                {
                    "step_delta": round(_cb(step_hint) if step_hint != 0.0 else 0.001, 4),
                    "fraction_errors_fixed": round(fraction_errors_fixed, 4),
                    "progress_ratio": round(progress_ratio, 4),
                    "step_budget_ratio": round(step_budget_ratio, 4),
                }
            )
            message = msg

        # Clamp ALL breakdown values — validator checks every numeric in API response
        breakdown = {k: round(_cb(v), 4) for k, v in breakdown.items()}
        reward = Reward(score=round(score, 4), breakdown=breakdown, message=message, done=done)
        obs = self._observation()
        info = {"valid_rows_deleted": valid_deleted}
        return obs, reward, done, info

    def _progress_ratio(self) -> float:
        if self.task is None or self.current_df is None:
            return 0.0
        remaining = len(detect_errors(self.task.task_id, self.current_df))
        return float(max(0.001, min(0.999, 1.0 - (remaining / max(1, self.total_errors_at_start)))))

    def state(self) -> EnvState:
        if self.task is None or self.current_df is None:
            return EnvState(ready=False)
        errs = detect_errors(self.task.task_id, self.current_df)
        return EnvState(
            ready=True,
            task_id=self.task.task_id,
            seed=self._episode_seed,
            step_count=self.step_count,
            max_steps=self.max_steps,
            done=self.done,
            rows=len(self.current_df),
            errors=errs,
            errors_remaining=len(errs),
            errors_initial=self.total_errors_at_start,
            progress_ratio=self._progress_ratio(),
        )
