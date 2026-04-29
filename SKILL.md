---
name: automl-pipeline
description: >
  Full-cycle AutoML pipeline in Python: load tabular data (.xlsx, .xls, .csv, .tsv),
  run Julius-style EDA, auto-engineer features, preprocess, train and cross-validate
  6 models, evaluate, show feature importance, and predict on new files.
  Trigger this skill whenever the user asks to: analyze a dataset, build a machine
  learning model, auto-train on Excel/CSV data, do EDA, predict on new data,
  run feature engineering, or says anything like "train a model on my data",
  "analyze this CSV", "make predictions from my Excel file", or "auto ML pipeline".
  Also trigger for: outlier detection, target correlation, skewness correction,
  one-hot encoding pipelines, or saving/loading sklearn models.
---

# AutoML Pipeline Skill

Builds a complete, self-contained Python AutoML script covering EDA → Feature
Engineering → Preprocessing → Model Selection → Evaluation → Prediction.

---

## 1. Understand the Request

Before writing code, confirm:

| Question | Default if not stated |
|---|---|
| Which file(s) to load? | Ask the user |
| Which column is the target? | Ask if not obvious |
| Classification or regression? | Auto-detect from target |
| Output prediction on a new file? | Only if user provides one |

If the user provides a file path and target column, proceed directly.

---

## 2. Script Architecture

Generate **one self-contained Python file** (e.g. `automl_pipeline.py`).
Use the section structure below, in order.

### Section 0 — Imports & Config

```python
import warnings, os, pickle, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              mean_absolute_error, mean_squared_error, r2_score)

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH   = "your_data.csv"          # ← replace or pass as sys.argv[1]
TARGET_COL  = "target"                 # ← replace or pass as sys.argv[2]
NEW_DATA    = None                     # ← path for offline prediction (optional)
MODEL_OUT   = "best_model.pkl"
META_OUT    = "model_meta.json"
```

---

### Section 1 — Data Loader

Support all four formats automatically:

```python
def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    loaders = {
        ".xlsx": lambda p: pd.read_excel(p, engine="openpyxl"),
        ".xls":  lambda p: pd.read_excel(p, engine="xlrd"),
        ".csv":  lambda p: pd.read_csv(p),
        ".tsv":  lambda p: pd.read_csv(p, sep="\t"),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported format: {ext}")
    df = loaders[ext](path)
    print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from {path}")
    return df
```

---

### Section 2 — Julius-Style EDA

Print a rich console report. Do **not** require matplotlib — text-only is fine unless the user asks for plots.

```python
def run_eda(df: pd.DataFrame, target: str) -> dict:
    print("\n" + "═"*60)
    print("  📊  EXPLORATORY DATA ANALYSIS")
    print("═"*60)

    # ── Column type detection ──────────────────────────────────
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    # Date detection: try parsing object cols
    date_cols = []
    for c in cat_cols[:]:
        try:
            parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.7:
                date_cols.append(c)
                cat_cols.remove(c)
        except Exception:
            pass

    print(f"\n🔢 Numeric   : {len(num_cols)}  cols → {num_cols[:8]}{'…' if len(num_cols)>8 else ''}")
    print(f"🔤 Categoric : {len(cat_cols)}  cols → {cat_cols[:8]}{'…' if len(cat_cols)>8 else ''}")
    print(f"📅 Date-like : {len(date_cols)} cols → {date_cols}")
    print(f"✅ Boolean   : {len(bool_cols)} cols → {bool_cols}")

    # ── Missing values ─────────────────────────────────────────
    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"\n⚠️  Missing values ({len(miss)} cols):")
    for col, pct in miss.head(10).items():
        bar = "█" * int(pct * 20)
        print(f"   {col:<30} {pct*100:5.1f}%  {bar}")

    # ── Duplicates ─────────────────────────────────────────────
    dupes = df.duplicated().sum()
    print(f"\n♻️  Duplicate rows : {dupes:,}  ({dupes/len(df)*100:.1f}%)")

    # ── Outliers (IQR) ─────────────────────────────────────────
    print("\n📌 Outliers (IQR method, top cols):")
    outlier_report = {}
    for c in num_cols[:15]:
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[c] < Q1 - 1.5*IQR) | (df[c] > Q3 + 1.5*IQR)).sum()
        outlier_report[c] = int(n_out)
        if n_out > 0:
            print(f"   {c:<30} {n_out:>5} outliers ({n_out/len(df)*100:.1f}%)")

    # ── Task type detection ────────────────────────────────────
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    n_unique = df[target].nunique()
    task = "classification" if (
        df[target].dtype == object or
        n_unique <= 20 or
        df[target].dtype == bool
    ) else "regression"
    print(f"\n🎯 Target   : '{target}'  →  {task.upper()}  ({n_unique} unique values)")

    # ── Top correlations with target ───────────────────────────
    print(f"\n🔗 Top correlations with target:")
    if task == "regression":
        corr = df[num_cols].corr()[target].drop(target, errors="ignore")
        corr = corr.abs().sort_values(ascending=False).head(10)
        for feat, val in corr.items():
            print(f"   {feat:<30} r = {val:.3f}")
    else:
        # Point-biserial / label-encoded correlation
        encoded_target = pd.Categorical(df[target]).codes
        corrs = {}
        for c in num_cols:
            if c == target: continue
            valid = df[c].dropna()
            idx   = valid.index.intersection(pd.Series(encoded_target).dropna().index)
            if len(idx) > 10:
                r, _ = stats.pearsonr(df.loc[idx, c].fillna(0), encoded_target[idx])
                corrs[c] = abs(r)
        corrs = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, val in corrs:
            print(f"   {feat:<30} r = {val:.3f}")

    print("═"*60)
    return {
        "num_cols": num_cols, "cat_cols": cat_cols,
        "date_cols": date_cols, "bool_cols": bool_cols,
        "task": task, "missing": miss.to_dict(), "outliers": outlier_report
    }
```

---

### Section 3 — Feature Engineering

Apply transformations **before** the sklearn pipeline (Pandas-level):

```python
def engineer_features(df: pd.DataFrame, eda: dict, target: str) -> pd.DataFrame:
    print("\n🔧 Feature Engineering…")
    df = df.copy()

    # 📅 Date columns → components
    for c in eda["date_cols"]:
        parsed = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}_year"]  = parsed.dt.year
        df[f"{c}_month"] = parsed.dt.month
        df[f"{c}_day"]   = parsed.dt.day
        df[f"{c}_dow"]   = parsed.dt.dayofweek
        df[f"{c}_week"]  = parsed.dt.isocalendar().week.astype(int)
        df.drop(columns=[c], inplace=True)
        print(f"   📅 Parsed date: {c} → year/month/day/dow/week")

    # 📐 Log-transform skewed numeric cols (skew > 1), skip target
    num_feats = [c for c in df.select_dtypes("number").columns if c != target]
    for c in num_feats:
        sk = df[c].skew()
        if abs(sk) > 1 and df[c].min() >= 0:
            df[f"{c}_log"] = np.log1p(df[c])
            print(f"   📐 Log-transform: {c}  (skew={sk:.2f})")

    # 🗑️ Drop high-missing (>60%) and zero-variance cols
    miss_pct = df.isnull().mean()
    drop_miss = miss_pct[miss_pct > 0.6].index.tolist()
    if drop_miss:
        df.drop(columns=drop_miss, inplace=True)
        print(f"   🗑️  Dropped high-missing cols: {drop_miss}")

    zero_var = [c for c in df.select_dtypes("number").columns
                if c != target and df[c].std(skipna=True) == 0]
    if zero_var:
        df.drop(columns=zero_var, inplace=True)
        print(f"   🗑️  Dropped zero-variance cols: {zero_var}")

    # 🔢 Frequency-encode high-cardinality categoricals (>20 uniques)
    for c in df.select_dtypes(["object", "category"]).columns:
        if c == target: continue
        if df[c].nunique() > 20:
            freq = df[c].value_counts(normalize=True)
            df[f"{c}_freq"] = df[c].map(freq)
            df.drop(columns=[c], inplace=True)
            print(f"   🔢 Freq-encoded high-cardinality: {c} ({freq.shape[0]} uniques)")

    # ✨ Interaction features from top-5 numeric pairs (correlation with target)
    num_feats2 = [c for c in df.select_dtypes("number").columns if c != target]
    if len(num_feats2) >= 2 and target in df.columns:
        try:
            if df[target].dtype == object:
                t_enc = pd.Categorical(df[target]).codes
            else:
                t_enc = df[target].fillna(0)
            corrs = {}
            for c in num_feats2:
                try:
                    r, _ = stats.pearsonr(df[c].fillna(0), t_enc)
                    corrs[c] = abs(r)
                except Exception:
                    pass
            top5 = sorted(corrs, key=corrs.get, reverse=True)[:5]
            count = 0
            for i in range(len(top5)):
                for j in range(i+1, len(top5)):
                    a, b = top5[i], top5[j]
                    df[f"{a}_x_{b}"] = df[a].fillna(0) * df[b].fillna(0)
                    count += 1
            print(f"   ✨ Created {count} interaction features from top-5 numeric cols")
        except Exception as e:
            print(f"   ⚠️  Interaction features skipped: {e}")

    print(f"   ✅ Shape after FE: {df.shape}")
    return df
```

---

### Section 4 — Preprocessing Pipeline

Build a `ColumnTransformer` after feature engineering:

```python
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = X.select_dtypes(["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols: transformers.append(("num", num_pipe, num_cols))
    if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")
```

---

### Section 5 — Model Selection (5-fold CV)

```python
CLASSIFIERS = {
    "RandomForest":      RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting":  GradientBoostingClassifier(random_state=42),
    "ExtraTrees":        ExtraTreesClassifier(n_estimators=200, random_state=42),
    "LogisticRegression":LogisticRegression(max_iter=500, random_state=42),
    "DecisionTree":      DecisionTreeClassifier(random_state=42),
    "KNN":               KNeighborsClassifier(),
}
REGRESSORS = {
    "RandomForest":      RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting":  GradientBoostingRegressor(random_state=42),
    "ExtraTrees":        ExtraTreesRegressor(n_estimators=200, random_state=42),
    "Ridge":             Ridge(),
    "DecisionTree":      DecisionTreeRegressor(random_state=42),
    "KNN":               KNeighborsRegressor(),
}

def select_model(X, y, preprocessor, task):
    print("\n🤖 Model Selection (5-fold CV)…")
    models   = CLASSIFIERS if task == "classification" else REGRESSORS
    cv       = StratifiedKFold(5, shuffle=True, random_state=42) if task == "classification" \
               else KFold(5, shuffle=True, random_state=42)
    scoring  = "roc_auc_ovr_weighted" if task == "classification" else "neg_root_mean_squared_error"

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            results[name] = scores.mean()
            sign   = "" if task == "classification" else "-"
            print(f"   {name:<22} CV score = {sign}{abs(scores.mean()):.4f} ± {scores.std():.4f}")
        except Exception as e:
            print(f"   {name:<22} ⚠️  Failed: {e}")

    best_name  = max(results, key=results.get) if task == "classification" \
                 else max(results, key=lambda k: results[k])  # neg RMSE, higher is better
    best_model = Pipeline([
        ("prep",  preprocessor),
        ("model", (CLASSIFIERS if task == "classification" else REGRESSORS)[best_name])
    ])
    best_model.fit(X, y)
    print(f"\n   🏆 Best model: {best_name}")
    return best_model, best_name, results
```

---

### Section 6 — Evaluation

```python
def evaluate(model, X, y, task):
    print("\n📈 Evaluation on full training data:")
    preds = model.predict(X)
    if task == "classification":
        acc = accuracy_score(y, preds)
        f1  = f1_score(y, preds, average="weighted", zero_division=0)
        try:
            proba = model.predict_proba(X)
            auc   = roc_auc_score(pd.get_dummies(y), proba,
                                  multi_class="ovr", average="weighted")
        except Exception:
            auc = float("nan")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   F1 Score : {f1:.4f}")
        print(f"   ROC-AUC  : {auc:.4f}")
        return {"accuracy": acc, "f1": f1, "auc": auc}
    else:
        mae  = mean_absolute_error(y, preds)
        rmse = mean_squared_error(y, preds, squared=False)
        r2   = r2_score(y, preds)
        print(f"   MAE  : {mae:.4f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   R²   : {r2:.4f}")
        return {"mae": mae, "rmse": rmse, "r2": r2}
```

---

### Section 7 — Feature Importance

```python
def show_feature_importance(model, X, top_n=15):
    print(f"\n🌟 Top-{top_n} Feature Importances:")
    try:
        raw_model = model.named_steps["model"]
        if not hasattr(raw_model, "feature_importances_"):
            print("   (Model does not expose feature importances)")
            return
        prep = model.named_steps["prep"]
        feat_names = []
        for name, trans, cols in prep.transformers_:
            if name == "num":
                feat_names.extend(cols)
            elif name == "cat":
                ohe = trans.named_steps["ohe"]
                feat_names.extend(ohe.get_feature_names_out(cols).tolist())
        importances = raw_model.feature_importances_
        pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
        for feat, imp in pairs[:top_n]:
            bar = "█" * int(imp * 100)
            print(f"   {feat:<35} {imp:.4f}  {bar}")
    except Exception as e:
        print(f"   ⚠️  Could not extract importances: {e}")
```

---

### Section 8 — Save Model & Metadata

```python
def save_model(model, meta: dict, model_path: str, meta_path: str):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n💾 Model saved  → {model_path}")
    print(f"💾 Metadata     → {meta_path}")
```

---

### Section 9 — Offline Prediction

```python
def predict_new(new_path: str, target: str, model_path: str, meta_path: str):
    print(f"\n🔮 Predicting on: {new_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    df_new = load_data(new_path)

    # Re-apply same FE steps described in meta
    # (For production, wrap FE in a sklearn custom transformer)
    # Here we do a best-effort: drop columns that didn't exist in training
    train_cols = meta.get("feature_cols", [])
    available  = [c for c in train_cols if c in df_new.columns]
    missing_c  = [c for c in train_cols if c not in df_new.columns]
    if missing_c:
        print(f"   ⚠️  Missing cols filled with NaN: {missing_c}")
        for c in missing_c:
            df_new[c] = np.nan
    X_new = df_new[train_cols]

    preds = model.predict(X_new)
    df_new["prediction"] = preds
    out_path = new_path.replace(".", "_predictions.")
    if out_path.endswith((".xlsx_predictions.", ".xls_predictions.")):
        out_path = out_path.rsplit(".", 1)[0] + ".xlsx"
    else:
        out_path = out_path.rsplit(".", 1)[0] + ".csv"
    df_new.to_csv(out_path, index=False) if out_path.endswith(".csv") \
        else df_new.to_excel(out_path, index=False)
    print(f"   ✅ Predictions saved → {out_path}")
    return df_new
```

---

### Section 10 — Main Orchestrator

```python
def main():
    import sys
    data_path  = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    target_col = sys.argv[2] if len(sys.argv) > 2 else TARGET_COL
    new_data   = sys.argv[3] if len(sys.argv) > 3 else NEW_DATA

    # ── Load ──────────────────────────────────────────────────
    df = load_data(data_path)

    # ── EDA ───────────────────────────────────────────────────
    eda = run_eda(df, target_col)
    task = eda["task"]

    # ── Feature Engineering ───────────────────────────────────
    df_fe = engineer_features(df, eda, target_col)

    # ── Split X / y ───────────────────────────────────────────
    X = df_fe.drop(columns=[target_col])
    y = df_fe[target_col]
    feature_cols = X.columns.tolist()

    # ── Preprocessor ──────────────────────────────────────────
    preprocessor = build_preprocessor(X)

    # ── Model Selection ───────────────────────────────────────
    best_model, best_name, cv_results = select_model(X, y, preprocessor, task)

    # ── Evaluation ────────────────────────────────────────────
    metrics = evaluate(best_model, X, y, task)

    # ── Feature Importance ────────────────────────────────────
    show_feature_importance(best_model, X)

    # ── Save ──────────────────────────────────────────────────
    meta = {
        "task": task, "target": target_col,
        "best_model": best_name, "metrics": metrics,
        "feature_cols": feature_cols, "cv_results": cv_results,
        "data_path": data_path
    }
    save_model(best_model, meta, MODEL_OUT, META_OUT)

    # ── Predict on new file ───────────────────────────────────
    if new_data:
        predict_new(new_data, target_col, MODEL_OUT, META_OUT)

    print("\n✅ Pipeline complete!")

if __name__ == "__main__":
    main()
```

---

## 3. Dependencies

Always include a `requirements.txt` alongside the script:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
openpyxl>=3.1      # .xlsx read/write
xlrd>=2.0          # .xls read
```

Install command to show the user:
```bash
pip install pandas numpy scikit-learn scipy openpyxl xlrd
```

---

## 4. Usage Instructions to Give the User

```bash
# Basic run
python automl_pipeline.py my_data.csv target_column

# With prediction on new file
python automl_pipeline.py train.csv churn new_customers.xlsx

# Predict-only (after model is saved)
python automl_pipeline.py  # set NEW_DATA in CONFIG block
```

---

## 5. Edge Cases & Guardrails

| Situation | Handling |
|---|---|
| Target not found | Raise clear `ValueError` |
| All columns missing >60% | Warn and exit gracefully |
| Single-class target | Warn, skip AUC |
| Non-numeric target in regression | Auto-switch to classification |
| New file missing columns | Fill with NaN, warn user |
| Date parsing fails | Skip silently, treat as categorical |
| Model CV all fail | Raise with helpful message |

---

## 6. Output Files

| File | Contents |
|---|---|
| `automl_pipeline.py` | Full pipeline script |
| `requirements.txt` | Dependencies |
| `best_model.pkl` | Saved sklearn Pipeline (generated at runtime) |
| `model_meta.json` | Task type, metrics, feature list (generated at runtime) |
| `*_predictions.csv/xlsx` | Predictions on new data (generated at runtime) |

Always create both `automl_pipeline.py` and `requirements.txt` together.
Present both files to the user on completion.

---

## 7. Quality Checklist (self-verify before presenting)

- [ ] All 10 sections present and in order
- [ ] `load_data` handles all 4 extensions
- [ ] EDA prints column types, missing %, duplicates, outliers, task type, correlations
- [ ] FE covers date parsing, log-transform, freq-encoding, drop rules, interactions
- [ ] Preprocessor uses `RobustScaler` + `MedianImputer` for numerics
- [ ] 6 models defined for both classification and regression
- [ ] CV uses `StratifiedKFold` for classification, `KFold` for regression
- [ ] Evaluation prints correct metrics per task type
- [ ] Feature importance guarded for non-tree models
- [ ] `predict_new` handles missing columns gracefully
- [ ] `main()` accepts CLI args
- [ ] `requirements.txt` included
