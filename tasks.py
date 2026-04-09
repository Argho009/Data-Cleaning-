import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TaskBundle:
    task_id: str
    description: str
    df: pd.DataFrame
    original_df: pd.DataFrame
    expected_df: pd.DataFrame
    valid_row_ids: set
    target_schema: Dict[str, str]
    metadata: Dict[str, Any]


def _random_state(seed: int) -> random.Random:
    return random.Random(seed)


def _random_date(rng: random.Random) -> pd.Timestamp:
    year = rng.choice([2022, 2023, 2024])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return pd.Timestamp(year=year, month=month, day=day)


def _zip_from_city(city: str) -> str:
    return {
        "San Francisco": "94105",
        "Los Angeles": "90001",
        "San Diego": "92101",
        "Sacramento": "95814",
        "Austin": "73301",
        "Dallas": "75001",
        "Houston": "77001",
    }.get(city, "00000")


def build_task_easy(seed: int = 42) -> TaskBundle:
    rng = _random_state(seed)
    rows = []
    for row_id in range(100):
        d = _random_date(rng)
        fmt = rng.choice(["%d/%m/%Y", "%m-%d-%y", "%Y/%m/%d"])
        rows.append(
            {
                "row_id": row_id,
                "sale_id": f"S{1000 + row_id}",
                "sale_date": d.strftime(fmt),
                "amount": round(rng.uniform(10, 500), 2),
                "product": rng.choice(["Widget", "Gadget", "Doohickey"]),
            }
        )

    df = pd.DataFrame(rows)
    dup_count = int(0.15 * len(df))
    dup_rows = df.sample(n=dup_count, random_state=seed).copy()
    dup_rows["row_id"] = range(1000, 1000 + dup_count)
    dirty = pd.concat([df, dup_rows], ignore_index=True)

    expected = df.copy()
    expected["sale_date"] = pd.to_datetime(expected["sale_date"], errors="coerce", format="mixed")
    expected["sale_date"] = expected["sale_date"].dt.strftime("%Y-%m-%d")
    expected = expected.drop_duplicates(subset=["sale_id", "sale_date", "amount", "product"]).reset_index(drop=True)

    description = "Fix mixed date formats and remove exact duplicate sales rows."
    target_schema = {
        "row_id": "int64",
        "sale_id": "object",
        "sale_date": "YYYY-MM-DD",
        "amount": "float64",
        "product": "object",
    }

    return TaskBundle(
        task_id="task_easy",
        description=description,
        df=dirty,
        original_df=dirty.copy(deep=True),
        expected_df=expected,
        valid_row_ids=set(df["row_id"].tolist()),
        target_schema=target_schema,
        metadata={},
    )


def build_task_medium(seed: int = 43) -> TaskBundle:
    rng = _random_state(seed)
    base = []
    cities = ["San Francisco", "Los Angeles", "San Diego", "Sacramento", "Austin", "Dallas", "Houston"]
    state_variants = {
        "CA": ["CA", "California", "Calif.", "ca"],
        "TX": ["TX", "Texas", "Tex.", "tx"],
    }
    for row_id in range(120):
        city = rng.choice(cities)
        canonical = "CA" if city in {"San Francisco", "Los Angeles", "San Diego", "Sacramento"} else "TX"
        state = rng.choice(state_variants[canonical])
        zip_code = _zip_from_city(city)
        if rng.random() < 0.2:
            city = None
        if rng.random() < 0.25:
            zip_code = None
        base.append(
            {
                "row_id": row_id,
                "customer_id": f"C{2000 + row_id}",
                "street": f"{rng.randint(100, 999)} Main St",
                "city": city,
                "state": state,
                "zip_code": zip_code,
            }
        )

    dirty = pd.DataFrame(base)
    expected = dirty.copy(deep=True)

    mode_city = dirty["city"].dropna().mode()
    fallback_city = mode_city.iloc[0] if not mode_city.empty else "San Francisco"
    expected["city"] = expected["city"].fillna(fallback_city)

    state_map = {
        "CA": "CA",
        "California": "CA",
        "Calif.": "CA",
        "ca": "CA",
        "TX": "TX",
        "Texas": "TX",
        "Tex.": "TX",
        "tx": "TX",
    }
    expected["state"] = expected["state"].map(lambda x: state_map.get(x, x))
    expected["zip_code"] = expected.apply(
        lambda row: _zip_from_city(row["city"]) if pd.isna(row["zip_code"]) else str(row["zip_code"]),
        axis=1,
    )

    target_schema = {
        "row_id": "int64",
        "customer_id": "object",
        "street": "object",
        "city": "object_non_null",
        "state": "2-letter_state_code",
        "zip_code": "5-digit_string",
    }
    description = "Fill null city/zip fields and standardize state values."
    return TaskBundle(
        task_id="task_medium",
        description=description,
        df=dirty,
        original_df=dirty.copy(deep=True),
        expected_df=expected,
        valid_row_ids=set(dirty["row_id"].tolist()),
        target_schema=target_schema,
        metadata={"fallback_city": fallback_city},
    )


def build_task_hard(seed: int = 44) -> TaskBundle:
    rng = _random_state(seed)
    records = []
    first_names = ["John", "Jane", "Alice", "Bob", "Carol", "David", "Priya", "Jon"]
    last_names = ["Smith", "Brown", "Johnson", "Miller", "Davis"]
    for row_id in range(140):
        fn = rng.choice(first_names[:-1])
        ln = rng.choice(last_names)
        dob = pd.Timestamp(year=rng.randint(1975, 2001), month=rng.randint(1, 12), day=rng.randint(1, 28))
        salary = rng.randint(45000, 140000)
        phone = f"({rng.randint(100,999)}) {rng.randint(100,999)}-{rng.randint(1000,9999)}"
        age = 2026 - dob.year
        records.append(
            {
                "row_id": row_id,
                "employee_name": f"{fn} {ln}",
                "dob": dob.strftime("%Y-%m-%d"),
                "salary": f"${salary:,}",
                "phone": phone,
                "age": age,
            }
        )

    df = pd.DataFrame(records)
    fuzzy_dups = df.sample(n=18, random_state=seed).copy()
    fuzzy_dups["row_id"] = range(2000, 2018)
    fuzzy_dups["employee_name"] = fuzzy_dups["employee_name"].map(
        lambda n: n.replace("John", "Jon").replace("Jane", "Jan")
    )
    dirty = pd.concat([df, fuzzy_dups], ignore_index=True)

    bad_salary_idx = dirty.sample(n=20, random_state=seed + 1).index
    dirty.loc[bad_salary_idx, "salary"] = dirty.loc[bad_salary_idx, "salary"].str.replace("0", "O", regex=False)

    bad_age_idx = dirty.sample(n=8, random_state=seed + 2).index
    dirty.loc[bad_age_idx[:4], "age"] = 7
    dirty.loc[bad_age_idx[4:], "age"] = 199

    phone_idx = dirty.sample(n=30, random_state=seed + 3).index
    dirty.loc[phone_idx, "phone"] = dirty.loc[phone_idx, "phone"].map(
        lambda p: re.sub(r"\D", "", p)
    )

    expected = df.copy(deep=True)
    expected["salary"] = expected["salary"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(int)
    expected["phone"] = expected["phone"].map(lambda p: re.sub(r"\D", "", p))
    expected["phone"] = expected["phone"].map(lambda s: f"({s[0:3]}) {s[3:6]}-{s[6:10]}")
    expected = expected[(expected["age"] >= 18) & (expected["age"] <= 90)].reset_index(drop=True)

    target_schema = {
        "row_id": "int64",
        "employee_name": "object",
        "dob": "YYYY-MM-DD",
        "salary": "int64_positive",
        "phone": "(XXX) XXX-XXXX",
        "age": "18_to_90",
    }
    description = "Resolve fuzzy duplicates, corrupted numeric salary fields, phone formats, and age outliers."
    return TaskBundle(
        task_id="task_hard",
        description=description,
        df=dirty,
        original_df=dirty.copy(deep=True),
        expected_df=expected,
        valid_row_ids=set(df["row_id"].tolist()),
        target_schema=target_schema,
        metadata={},
    )


_DEFAULT_SEEDS = {"task_easy": 42, "task_medium": 43, "task_hard": 44}


def build_task(task_id: str, seed: Optional[int] = None) -> TaskBundle:
    s = _DEFAULT_SEEDS.get(task_id, 42) if seed is None else int(seed)
    if task_id == "task_easy":
        return build_task_easy(seed=s)
    if task_id == "task_medium":
        return build_task_medium(seed=s)
    if task_id == "task_hard":
        return build_task_hard(seed=s)
    raise ValueError(f"Unknown task_id: {task_id}")


def detect_errors(task_id: str, df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    if task_id == "task_easy":
        parsed = pd.to_datetime(df["sale_date"], errors="coerce", format="mixed")
        bad_dates = int(parsed.isna().sum())
        if bad_dates:
            errors.append(f"Unparseable dates: {bad_dates}")
        duplicate_count = int(df.duplicated(subset=["sale_id", "sale_date", "amount", "product"]).sum())
        if duplicate_count:
            errors.append(f"Duplicate rows: {duplicate_count}")
    elif task_id == "task_medium":
        null_city = int(df["city"].isna().sum())
        if null_city:
            errors.append(f"Null city values: {null_city}")
        bad_state = int((~df["state"].astype(str).str.upper().isin(["CA", "TX"])).sum())
        if bad_state:
            errors.append(f"Non-canonical state values: {bad_state}")
        missing_zip = int(df["zip_code"].isna().sum())
        if missing_zip:
            errors.append(f"Missing zip_code values: {missing_zip}")
    elif task_id == "task_hard":
        bad_salary = int(df["salary"].astype(str).str.contains("O").sum())
        if bad_salary:
            errors.append(f"Corrupted salary values: {bad_salary}")
        bad_age = int(((df["age"] < 18) | (df["age"] > 90)).sum())
        if bad_age:
            errors.append(f"Age outliers: {bad_age}")
        bad_phone = int((~df["phone"].astype(str).str.match(r"^\(\d{3}\)\s\d{3}-\d{4}$").fillna(False)).sum())
        if bad_phone:
            errors.append(f"Non-standard phone formats: {bad_phone}")
    return errors


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def grade_task(task_id: str, current: pd.DataFrame, bundle: TaskBundle) -> Dict[str, float]:
    if task_id == "task_easy":
        parsed = pd.to_datetime(current["sale_date"], errors="coerce", format="mixed")
        date_acc = float((~parsed.isna()).mean())
        dedup_target = len(bundle.expected_df)
        dedup_diff = abs(len(current.drop_duplicates(subset=["sale_id", "sale_date", "amount", "product"])) - dedup_target)
        dedup_score = max(0.0, 1.0 - (dedup_diff / max(1, dedup_target)))
        return {"date_parse_accuracy": date_acc, "duplicate_quality": dedup_score}

    if task_id == "task_medium":
        null_fill = 1.0 - float(current["city"].isna().mean())
        canonical = current["state"].astype(str).str.upper().isin(["CA", "TX"])
        standard_acc = float(canonical.mean())
        zip_ok = current["zip_code"].astype(str).str.match(r"^\d{5}$").fillna(False)
        schema_score = float(((~current["city"].isna()) & canonical & zip_ok).mean())
        return {"null_fill_rate": null_fill, "standardization_accuracy": standard_acc, "schema_compliance": schema_score}

    if task_id == "task_hard":
        # Fuzzy dedup proxy: rows with same DOB and high name similarity should collapse.
        pairs = 0
        matched = 0
        rows = current[["employee_name", "dob"]].astype(str).to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i]["dob"] == rows[j]["dob"]:
                    pairs += 1
                    if _similar(_normalize_name(rows[i]["employee_name"]), _normalize_name(rows[j]["employee_name"])) > 0.9:
                        matched += 1
        dedup_f1 = 1.0 if pairs == 0 else max(0.0, 1.0 - (matched / pairs))
        salary_fix = 1.0 - float(current["salary"].astype(str).str.contains("O").mean())
        age_ok = float(((current["age"] >= 18) & (current["age"] <= 90)).mean())
        false_del = max(0, len(bundle.valid_row_ids) - len(set(current["row_id"]).intersection(bundle.valid_row_ids)))
        deletion_safety = max(0.0, 1.0 - (false_del / max(1, len(bundle.valid_row_ids))))
        return {
            "fuzzy_dedup_f1": dedup_f1,
            "numeric_fix_accuracy": salary_fix,
            "outlier_removal_quality": age_ok,
            "zero_false_deletion_score": deletion_safety,
        }
    raise ValueError(f"Unknown task_id: {task_id}")
