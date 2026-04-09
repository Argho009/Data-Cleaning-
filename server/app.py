import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from environment import Action, DataCleaningEnv, EnvState, Observation, Reward


app = FastAPI(title="Data Cleaning Agent", version="1.0.0")
env = DataCleaningEnv(max_steps=15)
custom_state: Dict[str, Any] = {
    "loaded": False,
    "df": None,
    "schema": {},
    "step_count": 0,
    "max_steps": 30,
    "done": False,
    "last_smart_clean_signature": "",
    "original_file_name": "",
}


class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    seed: Optional[int] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class CustomLoadRequest(BaseModel):
    raw_text: str
    input_format: str = "auto"
    file_name: str = ""
    target_schema: Dict[str, str] = Field(default_factory=dict)


class CustomActionRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class RecommendationRequest(BaseModel):
    target_column: str = ""
    objective: str = "auto"


ALL_ML_TECHNIQUES: Dict[str, List[str]] = {
    "classification": [
        "Logistic Regression", "Naive Bayes", "K-Nearest Neighbors", "Decision Tree",
        "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM",
        "CatBoost", "Support Vector Machine", "Neural Network (MLP)",
    ],
    "regression": [
        "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net",
        "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor",
        "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor", "SVR", "MLP Regressor",
    ],
    "clustering": [
        "KMeans", "MiniBatch KMeans", "DBSCAN", "HDBSCAN", "Gaussian Mixture Model",
        "Agglomerative Clustering", "Spectral Clustering", "BIRCH", "OPTICS",
    ],
    "time_series": [
        "ARIMA", "SARIMA", "Prophet", "Exponential Smoothing", "VAR",
        "XGBoost with lag features", "Random Forest with lag features", "LSTM", "GRU", "Temporal CNN",
    ],
    "dimensionality_reduction": [
        "PCA", "Kernel PCA", "Truncated SVD", "ICA", "t-SNE", "UMAP",
    ],
    "anomaly_detection": [
        "Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope", "Autoencoder",
    ],
}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Data Cleaning Agent - ML Studio</title>
  <style>
    :root {
      --bg: #070b1a;
      --bg2: #0c1631;
      --card: #101a33;
      --card2: #1a2650;
      --accent: #22d3ee;
      --accent2: #a78bfa;
      --accent3: #f472b6;
      --text: #eef2ff;
      --muted: #a5b4fc;
      --ok: #34d399;
      --warn: #f59e0b;
      --border: rgba(148, 163, 184, 0.28);
      --chip-bg: rgba(34, 211, 238, 0.14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(1000px 500px at -10% -10%, #312e81 0%, transparent 60%),
        radial-gradient(900px 500px at 110% -10%, #831843 0%, transparent 55%),
        linear-gradient(160deg, var(--bg), var(--bg2));
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      min-height: 100vh;
      padding: 20px;
    }
    h1, h2, h3 { margin: 0 0 10px 0; letter-spacing: 0.2px; }
    h1 { font-size: 1.9rem; }
    h3 { color: #c4b5fd; }
    .subtitle { color: var(--muted); margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card {
      background: linear-gradient(155deg, rgba(26, 38, 80, 0.92), rgba(16, 26, 51, 0.92));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 16px 34px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
      backdrop-filter: blur(5px);
    }
    .full { grid-column: 1 / -1; }
    textarea, input, select, button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(165, 180, 252, 0.35);
      background: rgba(10, 18, 40, 0.92);
      color: var(--text);
      padding: 10px;
      margin: 6px 0 10px 0;
    }
    textarea:focus, input:focus, select:focus {
      outline: none;
      border-color: #67e8f9;
      box-shadow: 0 0 0 3px rgba(103,232,249,0.18);
    }
    textarea { min-height: 110px; resize: vertical; }
    button {
      background: linear-gradient(90deg, var(--accent2), var(--accent), var(--accent3));
      color: #081226;
      font-weight: 700;
      border: none;
      cursor: pointer;
      transition: transform .12s ease, filter .2s ease, box-shadow .2s ease;
    }
    button:hover { filter: brightness(1.06); transform: translateY(-1px); box-shadow: 0 8px 24px rgba(34,211,238,0.22); }
    .ghost {
      background: linear-gradient(160deg, rgba(10,18,40,0.92), rgba(17,24,39,0.92));
      color: var(--text);
      border: 1px solid rgba(165,180,252,0.45);
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    pre {
      margin: 0;
      background: linear-gradient(180deg, rgba(2,6,23,0.88), rgba(10,18,40,0.88));
      border: 1px solid rgba(148,163,184,0.3);
      border-radius: 10px;
      color: #dbeafe;
      padding: 10px;
      max-height: 280px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .pill {
      display: inline-block;
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 999px;
      margin-right: 6px;
      background: var(--chip-bg);
      border: 1px solid rgba(34,211,238,0.45);
      color: #67e8f9;
    }
    .ok { color: var(--ok); font-weight: 600; }
    .warn { color: var(--warn); font-weight: 600; }
    .hero-badges { margin: 10px 0 18px 0; }
    .hero-badges .pill { margin-bottom: 6px; }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>Data Cleaning Agent - ML Studio</h1>
  <div class="subtitle">Upload or paste CSV, clean with guided options, run Auto Clean, and get ML algorithm suggestions.</div>
  <div class="hero-badges">
    <span class="pill">Auto Detect CSV/JSON</span>
    <span class="pill">Beginner Friendly</span>
    <span class="pill">One-Click Smart Clean</span>
    <span class="pill">ML Technique Suggestions</span>
  </div>

  <div class="grid">
    <section class="card">
      <h3>1) Dataset Input</h3>
      <label class="pill">CSV/JSON Upload (auto detect)</label>
      <input id="fileInput" type="file" accept=".csv,.json" />
      <label class="pill">Or Paste CSV/JSON</label>
      <textarea id="csvText" placeholder='CSV: col1,col2,target&#10;1,9,A&#10;2,,B&#10;&#10;JSON: [{"col1":1,"target":"A"}]'></textarea>
      <label class="pill">Target Schema JSON (optional)</label>
      <textarea id="schemaText" placeholder='{"target":"object","feature_1":"float64"}'></textarea>
      <button onclick="loadCsv()">Load Dataset (Auto Detect)</button>
      <pre id="loadOut">Waiting for dataset...</pre>
    </section>

    <section class="card">
      <h3>2) Guided Cleaning Actions</h3>
      <div class="row">
        <div>
          <label class="pill">Action</label>
          <select id="actionType">
            <option>remove_duplicates</option>
            <option>fill_nulls</option>
            <option>fix_dates</option>
            <option>fix_numeric</option>
            <option>standardize</option>
            <option>remove_outliers</option>
            <option>encode_categorical</option>
            <option>scale_numeric</option>
            <option>semi_auto_clean</option>
            <option>inspect</option>
            <option>submit</option>
          </select>
        </div>
        <div>
          <label class="pill">Null Strategy</label>
          <select id="nullStrategy">
            <option>mode</option>
            <option>mean</option>
            <option>median</option>
            <option>constant</option>
          </select>
        </div>
      </div>
      <label class="pill">Columns (comma-separated, optional)</label>
      <input id="columnsText" placeholder="age,salary,state" />
      <label class="pill">Constant fill value (used if strategy=constant)</label>
      <input id="constantValue" placeholder="UNKNOWN or 0" />
      <label class="pill">Outlier Column (optional)</label>
      <input id="outlierColumn" placeholder="age" />
      <label class="pill">Cleaning Mode</label>
      <select id="cleanMode">
        <option>automatic</option>
        <option>semi_automatic</option>
      </select>
      <div class="row">
        <button onclick="runAction()">Run Action</button>
        <button class="ghost" onclick="smartClean()">Smart Clean</button>
      </div>
      <pre id="actionOut">No actions yet.</pre>
    </section>

    <section class="card full">
      <h3>3) ML Suggestions + Dataset Health</h3>
      <div class="row">
        <div>
          <label class="pill">Target Column (optional)</label>
          <input id="targetColumn" placeholder="target" />
        </div>
        <div>
          <label class="pill">Objective</label>
          <select id="objective">
            <option>auto</option>
            <option>classification</option>
            <option>regression</option>
            <option>clustering</option>
            <option>time_series</option>
          </select>
        </div>
      </div>
      <div class="row">
        <button onclick="getRecommendations()">Suggest Techniques (names only)</button>
        <button class="ghost" onclick="refreshState()">Refresh State</button>
      </div>
      <button class="ghost" onclick="downloadCleaned()">Download Cleaned Dataset</button>
      <label class="pill">Technical Details</label>
      <select id="showRaw">
        <option value="no">Hidden (simple view)</option>
        <option value="yes">Show raw JSON</option>
      </select>
      <div class="row">
        <pre id="recoOut">No recommendations yet. Press Enter in target field.</pre>
        <pre id="stateOut">Dataset summary will appear here.</pre>
      </div>
      <pre id="rawOut" style="display:none;">Raw details hidden.</pre>
    </section>
  </div>

  <script>
    const fileInput = document.getElementById("fileInput");
    let loadedFileName = "";
    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      loadedFileName = file.name || "";
      const text = await file.text();
      document.getElementById("csvText").value = text;
    });

    function parseColumns(raw) {
      return raw.split(",").map(x => x.trim()).filter(Boolean);
    }

    async function loadCsv() {
      try {
        const raw_text = document.getElementById("csvText").value;
        let target_schema = {};
        const rawSchema = document.getElementById("schemaText").value.trim();
        if (rawSchema) target_schema = JSON.parse(rawSchema);
        const res = await fetch("/ui/load_csv", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({ raw_text, input_format: "auto", file_name: loadedFileName, target_schema })
        });
        const data = await res.json();
        document.getElementById("loadOut").textContent =
          "Loaded successfully\\n" +
          "Format: " + (data.detected_format || "auto") + "\\n" +
          "Rows: " + (data.rows || 0) + "\\n" +
          "Columns: " + ((data.columns || []).length);
        updateRawPanel(data);
        await refreshState();
      } catch (err) {
        document.getElementById("loadOut").textContent = "Load failed: " + err;
      }
    }

    async function runAction() {
      const action_type = document.getElementById("actionType").value;
      const columns = parseColumns(document.getElementById("columnsText").value);
      const strategy = document.getElementById("nullStrategy").value;
      const value = document.getElementById("constantValue").value;
      const outlierCol = document.getElementById("outlierColumn").value.trim();
      const parameters = {};
      if (columns.length) parameters.columns = columns;
      if (action_type === "fill_nulls") {
        parameters.strategy = strategy;
        if (strategy === "constant") parameters.value = value;
      }
      if (action_type === "remove_outliers" && outlierCol) parameters.column = outlierCol;
      if (action_type === "semi_auto_clean") {
        parameters.remove_duplicates = true;
        parameters.fill_strategy = strategy;
        parameters.fill_value = value || "UNKNOWN";
        parameters.remove_outliers = !!outlierCol;
        parameters.outlier_column = outlierCol;
      }

      const res = await fetch("/ui/action", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ action_type, parameters })
      });
      const data = await res.json();
      const beforeRows = data.before_rows ?? "-";
      const afterRows = data.after_rows ?? data.rows ?? "-";
      document.getElementById("actionOut").textContent =
        (data.message || "Action complete.") + "\\n" +
        "Action: " + (data.action || action_type) + "\\n" +
        "Changed: " + (data.changed === true ? "Yes" : (data.changed === false ? "No" : "-")) + "\\n" +
        "Before rows: " + beforeRows + "\\n" +
        "After rows: " + afterRows;
      updateRawPanel(data);
      await refreshState();
    }

    async function smartClean() {
      const mode = document.getElementById("cleanMode").value;
      const strategy = document.getElementById("nullStrategy").value;
      const outlierCol = document.getElementById("outlierColumn").value.trim();
      const action_type = mode === "automatic" ? "auto_clean" : "semi_auto_clean";
      const parameters = mode === "automatic" ? {} : {
        remove_duplicates: true,
        fill_strategy: strategy,
        remove_outliers: !!outlierCol,
        outlier_column: outlierCol
      };
      const res = await fetch("/ui/action", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ action_type, parameters })
      });
      const data = await res.json();
      document.getElementById("actionOut").textContent =
        (data.message || "Smart clean completed") + "\\n" +
        "Changed: " + (data.changed === true ? "Yes" : "No") + "\\n" +
        "Before rows: " + (data.before_rows ?? "-") + "\\n" +
        "After rows: " + (data.after_rows ?? data.rows ?? "-");
      updateRawPanel(data);
      await refreshState();
    }

    async function getRecommendations() {
      const target_column = document.getElementById("targetColumn").value.trim();
      const objective = document.getElementById("objective").value;
      const res = await fetch("/ui/recommendations", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ target_column, objective })
      });
      const data = await res.json();
      if (Array.isArray(data.technique_scores) && data.technique_scores.length > 0) {
        document.getElementById("recoOut").textContent =
          data.technique_scores.map(x => `${x.technique} - ${x.fit_percent}%`).join("\\n");
      } else {
        document.getElementById("recoOut").textContent = (data.technique_names || []).join("\\n");
      }
    }

    async function downloadCleaned() {
      window.open("/ui/download?format=csv", "_blank");
    }

    document.getElementById("targetColumn").addEventListener("keydown", async (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        await getRecommendations();
      }
    });

    async function refreshState() {
      const res = await fetch("/ui/summary");
      const data = await res.json();
      if (!data.loaded) {
        document.getElementById("stateOut").textContent = "Load a dataset to see summary.";
        return;
      }
      const s = data.summary || {};
      const lines = [];
      lines.push("Rows: " + (s.rows || 0));
      lines.push("Columns: " + (s.columns || 0));
      lines.push("Memory: " + (s.memory_mb || 0) + " MB");
      lines.push("Issues found: " + ((data.errors || []).length));
      const topNull = Object.entries(s.null_count_by_column || {}).filter(([_,v]) => v > 0).slice(0, 8);
      if (topNull.length) {
        lines.push("");
        lines.push("Missing values:");
        for (const [k,v] of topNull) lines.push("- " + k + ": " + v);
      }
      const dtypes = Object.entries(s.dtypes || {}).slice(0, 8);
      if (dtypes.length) {
        lines.push("");
        lines.push("Column types:");
        for (const [k,v] of dtypes) lines.push("- " + k + ": " + v);
      }
      document.getElementById("stateOut").textContent = lines.join("\\n");
      updateRawPanel(data);
    }

    function updateRawPanel(payload) {
      const raw = document.getElementById("rawOut");
      const show = document.getElementById("showRaw").value === "yes";
      raw.style.display = show ? "block" : "none";
      if (show) raw.textContent = JSON.stringify(payload, null, 2);
    }
  </script>
</body>
</html>
"""


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = Body(default=None)) -> Observation:
    payload = req or ResetRequest()
    try:
        return env.reset(payload.task_id, seed=payload.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    return env.state()


def _custom_error_report(df: pd.DataFrame, schema: Dict[str, str]) -> List[str]:
    errors: List[str] = []
    if df.empty:
        return ["Dataset is empty."]
    dup_count = int(df.duplicated().sum())
    if dup_count:
        errors.append(f"Duplicate rows: {dup_count}")
    null_count = int(df.isna().sum().sum())
    if null_count:
        errors.append(f"Null cells: {null_count}")
    for col in df.columns:
        if "date" in col.lower():
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            bad = int(parsed.isna().sum())
            if bad:
                errors.append(f"Unparseable dates in '{col}': {bad}")
    if schema:
        missing = [k for k in schema.keys() if k not in df.columns]
        if missing:
            errors.append(f"Missing schema columns: {missing}")
    return errors


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("O", "0", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    for col in out.columns:
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="float")
        elif out[col].dtype == "object":
            unique_ratio = float(out[col].nunique(dropna=True)) / max(1, len(out[col]))
            if unique_ratio < 0.5:
                out[col] = out[col].astype("category")
    return out


def _parse_input_to_df(raw_text: str, input_format: str, file_name: str) -> pd.DataFrame:
    fmt = (input_format or "auto").lower()
    name = (file_name or "").lower()
    text = raw_text or ""
    if not text.strip():
        raise ValueError("Input text is empty.")

    if fmt == "auto":
        if name.endswith(".json") or text.lstrip().startswith("[") or text.lstrip().startswith("{"):
            fmt = "json"
        else:
            fmt = "csv"

    if fmt == "json":
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if all(isinstance(v, list) for v in parsed.values()):
                return pd.DataFrame(parsed)
            return pd.DataFrame([parsed])
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        raise ValueError("Unsupported JSON structure. Use object, list of objects, or column->list map.")
    if fmt == "csv":
        return pd.read_csv(io.StringIO(text))
    raise ValueError("Unsupported input_format. Use auto, csv, or json.")


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    for col in out.columns:
        if "date" in col.lower():
            parsed = pd.to_datetime(out[col], errors="coerce", dayfirst=True)
            out[col] = parsed.dt.strftime("%Y-%m-%d")
    return out


def _fix_dates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy(deep=True)
    cols = columns or [c for c in out.columns if "date" in c.lower()]
    for col in cols:
        if col in out.columns:
            parsed = pd.to_datetime(out[col], errors="coerce", dayfirst=True)
            out[col] = parsed.dt.strftime("%Y-%m-%d")
    return out


def _fill_nulls(df: pd.DataFrame, strategy: str, columns: List[str], value: Any = "UNKNOWN") -> pd.DataFrame:
    out = df.copy(deep=True)
    cols = columns or [c for c in out.columns if out[c].isna().any()]
    for col in cols:
        if col not in out.columns:
            continue
        if strategy == "constant":
            fill_value = value
        elif strategy in {"mean", "median"}:
            numeric = pd.to_numeric(out[col], errors="coerce")
            if strategy == "mean":
                fill_value = float(numeric.mean()) if not numeric.dropna().empty else 0.0
            else:
                fill_value = float(numeric.median()) if not numeric.dropna().empty else 0.0
            out[col] = numeric
        else:
            mode = out[col].dropna().mode()
            fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
        out[col] = out[col].fillna(fill_value)
    return out


def _standardize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy(deep=True)
    cols = columns or out.columns.tolist()
    state_map = {
        "CA": "CA", "California": "CA", "Calif.": "CA", "ca": "CA",
        "TX": "TX", "Texas": "TX", "Tex.": "TX", "tx": "TX",
    }
    for col in cols:
        if col not in out.columns:
            continue
        if col.lower() == "state":
            out[col] = out[col].map(lambda x: state_map.get(x, x))
        if col.lower() == "phone":
            out[col] = out[col].astype(str).map(
                lambda p: (lambda digits: f"({digits[-10:-7]}) {digits[-7:-4]}-{digits[-4:]}"
                           if len(digits) >= 10 else p)(re.sub(r"\D", "", p))
            )
    return out


def _remove_outliers_iqr(df: pd.DataFrame, column: str = "") -> pd.DataFrame:
    out = df.copy(deep=True)
    if column:
        columns = [column] if column in out.columns else []
    else:
        columns = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for col in columns:
        vals = pd.to_numeric(out[col], errors="coerce")
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out = out[(vals >= lower) & (vals <= upper) | vals.isna()]
    return out.reset_index(drop=True)


def _encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy(deep=True)
    cols = columns or [c for c in out.columns if out[c].dtype == "object"]
    valid = [c for c in cols if c in out.columns]
    if not valid:
        return out
    return pd.get_dummies(out, columns=valid, drop_first=False)


def _scale_numeric(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    out = df.copy(deep=True)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for col in numeric_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        if method == "minmax":
            min_v = s.min()
            max_v = s.max()
            out[col] = 0.0 if pd.isna(min_v) or max_v == min_v else (s - min_v) / (max_v - min_v)
        else:
            mean = s.mean()
            std = s.std()
            out[col] = 0.0 if pd.isna(std) or std == 0 else (s - mean) / std
    return out


def _auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    out = out.drop_duplicates().reset_index(drop=True)
    out = _try_parse_dates(out)
    for col in out.columns:
        if out[col].dtype == "object":
            numeric_candidate = _coerce_numeric_series(out[col])
            ratio = float(numeric_candidate.notna().mean())
            if ratio > 0.8:
                out[col] = numeric_candidate
    out = _fill_nulls(out, strategy="median", columns=[c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])])
    out = _fill_nulls(
        out,
        strategy="mode",
        columns=[c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])],
    )
    out = _standardize(out, columns=out.columns.tolist())
    out = _remove_outliers_iqr(out)
    out = _optimize_memory(out)
    return out.reset_index(drop=True)


def _semi_auto_clean(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy(deep=True)
    if params.get("remove_duplicates", True):
        out = out.drop_duplicates().reset_index(drop=True)
    out = _fix_dates(out, columns=params.get("date_columns", []))
    fill_strategy = params.get("fill_strategy", "mode")
    fill_columns = params.get("fill_columns", [])
    fill_value = params.get("fill_value", "UNKNOWN")
    out = _fill_nulls(out, strategy=fill_strategy, columns=fill_columns, value=fill_value)
    if params.get("fix_numeric", True):
        for col in out.columns:
            if out[col].dtype == "object":
                numeric_candidate = _coerce_numeric_series(out[col])
                if float(numeric_candidate.notna().mean()) > 0.85:
                    out[col] = numeric_candidate
    if params.get("remove_outliers", False):
        out = _remove_outliers_iqr(out, column=params.get("outlier_column", ""))
    out = _optimize_memory(out)
    return out.reset_index(drop=True)


def _dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    nulls = (df.isna().sum() / max(1, len(df))).round(4).to_dict()
    dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "null_ratio": nulls,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def _dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 4)
    dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    nulls = df.isna().sum().to_dict()
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    describe_table = {}
    if numeric:
        describe_df = df[numeric].describe().round(3)
        describe_table = json.loads(describe_df.to_json())
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "memory_mb": memory_mb,
        "dtypes": dtypes,
        "null_count_by_column": nulls,
        "describe_numeric": describe_table,
    }


def _df_signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    row_hash = pd.util.hash_pandas_object(df, index=True).astype("uint64")
    return f"{len(df)}|{len(df.columns)}|{int(row_hash.sum())}"


def _regression_scores(df: pd.DataFrame) -> List[Dict[str, Any]]:
    profile = _dataset_profile(df)
    rows = profile["rows"]
    n_features = max(1, len(profile["columns"]) - 1)
    null_penalty = sum(profile["null_ratio"].values()) / max(1, len(profile["null_ratio"]))
    numeric_ratio = len(profile["numeric_columns"]) / max(1, len(profile["columns"]))

    complexity = min(1.0, (n_features / 30.0))
    size_bonus = min(1.0, rows / 5000.0)
    cleanliness = max(0.0, 1.0 - min(1.0, null_penalty))

    base = {
        "Linear Regression": 68,
        "Ridge Regression": 72,
        "Lasso Regression": 70,
        "Elastic Net": 73,
        "Decision Tree Regressor": 74,
        "Random Forest Regressor": 82,
        "Gradient Boosting Regressor": 83,
        "XGBoost Regressor": 86,
        "LightGBM Regressor": 85,
        "CatBoost Regressor": 84,
        "SVR": 76,
        "MLP Regressor": 78,
    }

    scores: List[Dict[str, Any]] = []
    for name, start in base.items():
        score = float(start)
        if "Linear" in name or "Ridge" in name or "Lasso" in name or "Elastic Net" in name:
            score += (numeric_ratio * 5.0) + (cleanliness * 6.0) - (complexity * 3.0)
        elif "Tree" in name or "Forest" in name or "Boost" in name or "XGBoost" in name or "LightGBM" in name or "CatBoost" in name:
            score += (complexity * 6.0) + (size_bonus * 5.0) + (cleanliness * 4.0)
        elif "SVR" in name:
            score += (numeric_ratio * 4.0) + (cleanliness * 5.0) - (complexity * 2.0)
        else:  # MLP
            score += (size_bonus * 6.0) + (complexity * 3.0) + (cleanliness * 2.0)
        score = max(55.0, min(97.0, score))
        scores.append({"technique": name, "fit_percent": round(score, 1)})

    scores.sort(key=lambda x: x["fit_percent"], reverse=True)
    return scores


def _recommend_models(df: pd.DataFrame, target_column: str, objective: str) -> Dict[str, Any]:
    profile = _dataset_profile(df)
    inferred_objective = objective
    if objective == "auto":
        if target_column and target_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                unique = int(df[target_column].nunique(dropna=True))
                inferred_objective = "classification" if 2 <= unique <= 20 else "regression"
            else:
                inferred_objective = "classification"
        elif any("date" in c.lower() for c in df.columns):
            inferred_objective = "time_series"
        else:
            inferred_objective = "clustering"

    suggestions = {
        "classification": ALL_ML_TECHNIQUES["classification"],
        "regression": ALL_ML_TECHNIQUES["regression"],
        "clustering": ALL_ML_TECHNIQUES["clustering"],
        "time_series": ALL_ML_TECHNIQUES["time_series"],
    }
    optimization_tips = [
        "Use train/validation split or cross-validation.",
        "Tune hyperparameters with random search first.",
        "Handle class imbalance with class weights or resampling.",
        "Track metrics that match your objective (F1/AUC, RMSE/MAE).",
    ]
    return {
        "objective": inferred_objective,
        "candidate_algorithms": suggestions.get(inferred_objective, []),
        "recommended_preprocessing": [
            "Deduplicate rows",
            "Impute null values (median for numeric, mode for categorical)",
            "Encode categorical features",
            "Scale numeric features",
        ],
        "optimization_tips": optimization_tips,
        "profile": profile,
        "target_column_used": target_column or None,
    }


def _cleaned_download_filename(original_file_name: str, output_ext: str) -> str:
    """Build e.g. `sales (cleaned).csv` from uploaded `sales.csv`."""
    name = (original_file_name or "").strip() or "dataset"
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    if not stem:
        stem = "dataset"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    stem = stem.strip() or "dataset"
    ext = output_ext.lstrip(".").lower()
    return f"{stem} (cleaned).{ext}"


@app.post("/ui/load_csv")
def ui_load_csv(req: CustomLoadRequest) -> Dict[str, Any]:
    try:
        df = _parse_input_to_df(req.raw_text, req.input_format, req.file_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {exc}") from exc
    df = _optimize_memory(df)
    custom_state["loaded"] = True
    custom_state["df"] = df
    custom_state["schema"] = req.target_schema or {}
    custom_state["step_count"] = 0
    custom_state["done"] = False
    custom_state["last_smart_clean_signature"] = ""
    custom_state["original_file_name"] = (req.file_name or "").strip()
    detected = "json" if (req.file_name.lower().endswith(".json") or req.raw_text.lstrip().startswith(("[", "{"))) else "csv"
    return {"ok": True, "detected_format": detected, "rows": len(df), "columns": df.columns.tolist(), "profile": _dataset_profile(df)}


@app.post("/ui/action")
def ui_action(req: CustomActionRequest) -> Dict[str, Any]:
    if not custom_state["loaded"] or custom_state["df"] is None:
        raise HTTPException(status_code=400, detail="Load CSV first.")
    if custom_state["done"]:
        raise HTTPException(status_code=400, detail="Session already submitted. Reload CSV to start again.")

    before = custom_state["df"]
    custom_state["step_count"] += 1
    action_type = req.action_type
    params = req.parameters or {}

    if action_type == "submit":
        custom_state["done"] = True
        errors = _custom_error_report(before, custom_state["schema"])
        score = max(0.0, min(1.0, 1.0 - (len(errors) / 10.0)))
        return {"done": True, "score": round(score, 4), "message": "Submitted custom dataset.", "errors_remaining": errors}

    before_rows = len(before)
    out = before.copy(deep=True)
    columns = params.get("columns", [])
    if action_type == "inspect":
        pass
    elif action_type == "remove_duplicates":
        subset = params.get("subset")
        out = out.drop_duplicates(subset=subset).reset_index(drop=True) if subset else out.drop_duplicates().reset_index(drop=True)
    elif action_type == "fill_nulls":
        out = _fill_nulls(
            out,
            strategy=params.get("strategy", "mode"),
            columns=columns,
            value=params.get("value", "UNKNOWN"),
        )
    elif action_type == "fix_dates":
        out = _fix_dates(out, columns=columns)
    elif action_type == "fix_numeric":
        cols = columns or out.columns.tolist()
        for col in cols:
            if col in out.columns:
                out[col] = _coerce_numeric_series(out[col])
    elif action_type == "standardize":
        out = _standardize(out, columns=columns)
    elif action_type == "remove_outliers":
        out = _remove_outliers_iqr(out, column=params.get("column", ""))
    elif action_type == "encode_categorical":
        out = _encode_categorical(out, columns=columns)
    elif action_type == "scale_numeric":
        out = _scale_numeric(out, method=params.get("method", "standard"))
    elif action_type == "auto_clean":
        sig_before = _df_signature(before)
        if custom_state.get("last_smart_clean_signature") == sig_before:
            return {
                "done": False,
                "action": action_type,
                "changed": False,
                "before_rows": before_rows,
                "after_rows": before_rows,
                "message": "Dataset already cleaned. No extra changes applied.",
            }
        out = _auto_clean(out)
    elif action_type == "semi_auto_clean":
        sig_before = _df_signature(before)
        if custom_state.get("last_smart_clean_signature") == sig_before:
            return {
                "done": False,
                "action": action_type,
                "changed": False,
                "before_rows": before_rows,
                "after_rows": before_rows,
                "message": "Dataset already cleaned. No extra changes applied.",
            }
        out = _semi_auto_clean(out, params)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported custom action: {action_type}")

    custom_state["df"] = _optimize_memory(out.reset_index(drop=True))
    if action_type in {"auto_clean", "semi_auto_clean"}:
        custom_state["last_smart_clean_signature"] = _df_signature(custom_state["df"])
    changed = not before.equals(custom_state["df"])
    step_score = 0.05 if changed else -0.02
    return {
        "done": False,
        "action": action_type,
        "step_score": round(step_score, 4),
        "changed": changed,
        "before_rows": before_rows,
        "after_rows": len(custom_state["df"]),
        "rows": len(custom_state["df"]),
        "columns": custom_state["df"].columns.tolist(),
        "profile": _dataset_profile(custom_state["df"]),
        "message": "Clean dataset complete." if action_type in {"auto_clean", "semi_auto_clean"} else "Action complete.",
    }


@app.get("/ui/state")
def ui_state() -> Dict[str, Any]:
    if not custom_state["loaded"] or custom_state["df"] is None:
        return {"loaded": False}
    df = custom_state["df"]
    return {
        "loaded": True,
        "step_count": custom_state["step_count"],
        "max_steps": custom_state["max_steps"],
        "done": custom_state["done"],
        "rows": len(df),
        "columns": df.columns.tolist(),
        "errors": _custom_error_report(df, custom_state["schema"]),
        "profile": _dataset_profile(df),
        "preview": json.loads(df.head(10).to_json(orient="records")),
        "schema": custom_state["schema"],
    }


@app.get("/ui/summary")
def ui_summary() -> Dict[str, Any]:
    if not custom_state["loaded"] or custom_state["df"] is None:
        return {"loaded": False}
    df = custom_state["df"]
    return {
        "loaded": True,
        "summary": _dataset_summary(df),
        "errors": _custom_error_report(df, custom_state["schema"]),
    }


@app.post("/ui/recommendations")
def ui_recommendations(req: RecommendationRequest) -> Dict[str, Any]:
    if not custom_state["loaded"] or custom_state["df"] is None:
        raise HTTPException(status_code=400, detail="Load CSV first.")
    rec = _recommend_models(custom_state["df"], req.target_column, req.objective)
    response = {
        "objective": rec["objective"],
        "technique_names": rec["candidate_algorithms"],
    }
    if rec["objective"] == "regression":
        response["technique_scores"] = _regression_scores(custom_state["df"])
    return response


@app.get("/ui/ml-techniques")
def ui_ml_techniques() -> Dict[str, Any]:
    return {
        "all_ml_techniques": ALL_ML_TECHNIQUES,
        "total_technique_count": sum(len(v) for v in ALL_ML_TECHNIQUES.values()),
    }


@app.get("/ui/download")
def ui_download(format: str = "csv") -> StreamingResponse:
    if not custom_state["loaded"] or custom_state["df"] is None:
        raise HTTPException(status_code=400, detail="Load CSV first.")
    df = custom_state["df"]
    fmt = (format or "csv").lower()
    orig = custom_state.get("original_file_name") or ""
    if fmt == "json":
        payload = df.to_json(orient="records", indent=2)
        fname = _cleaned_download_filename(orig, "json")
        return StreamingResponse(
            io.BytesIO(payload.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    payload = df.to_csv(index=False)
    fname = _cleaned_download_filename(orig, "csv")
    return StreamingResponse(
        io.BytesIO(payload.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def main() -> None:
    import uvicorn
    # Use server.app:app since it's now in the server folder, wait, 
    # if python runs `python server/app.py`, the module name might be `__main__`.
    # Let's just use `uvicorn.run("server.app:app", ...)` assuming it runs from root.
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
