"""Build date-safe ensemble predictions for the static GitHub Pages site."""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "matches.json"
OUTPUT = ROOT / "docs" / "predictions.json"
FEATURES = [f"odds_{i}" for i in range(36, 75)]
TARGETS = {"2,5 Üst": "target_o25", "İY 1,5 Üst": "target_ht15", "KG Var": "target_btts"}
MS_LABELS = {1: "MS 1", 0: "MS X", 2: "MS 2"}
warnings.filterwarnings("ignore", category=UserWarning)


def fit_ensemble(train, target, multiclass=False):
    sample = train.dropna(subset=[target])
    if len(sample) < 30 or sample[target].nunique() < 2:
        return []

    classes = sorted(sample[target].astype(int).unique().tolist())
    encoded = sample[target].astype(int).map({value: i for i, value in enumerate(classes)})
    X = sample[FEATURES]
    n_classes = len(classes)
    xgb_args = dict(n_estimators=180, max_depth=5, learning_rate=0.045,
                    subsample=0.85, colsample_bytree=0.85, reg_lambda=2,
                    objective="multi:softprob" if multiclass else "binary:logistic",
                    eval_metric="mlogloss" if multiclass else "logloss",
                    n_jobs=2, random_state=42)
    lgb_args = dict(n_estimators=180, learning_rate=0.045, num_leaves=31,
                    objective="multiclass" if multiclass else "binary",
                    class_weight="balanced", verbosity=-1, n_jobs=2, random_state=42)
    if multiclass:
        xgb_args["num_class"] = n_classes
        lgb_args["num_class"] = n_classes

    estimators = [
        RandomForestClassifier(n_estimators=180, max_features="sqrt", min_samples_leaf=3,
                               class_weight="balanced_subsample", n_jobs=2, random_state=42),
        XGBClassifier(**xgb_args),
        LGBMClassifier(**lgb_args),
        CatBoostClassifier(iterations=180, depth=6, learning_rate=0.045, verbose=False,
                           loss_function="MultiClass" if multiclass else "Logloss",
                           auto_class_weights="Balanced", random_seed=42, thread_count=2),
    ]
    fitted = []
    for estimator in estimators:
        pipeline = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)
        fitted.append((pipeline.fit(X, encoded), classes))
    return fitted


def average_probabilities(models, X):
    if not models:
        return {}
    classes = sorted(set().union(*(set(original_classes) for _, original_classes in models)))
    return {
        label: np.mean([
            model.predict_proba(X)[:, original_classes.index(label)]
            if label in original_classes else np.zeros(len(X))
            for model, original_classes in models
        ], axis=0)
        for label in classes
    }


def outcome_result(row, label):
    if label.startswith("MS "):
        if pd.isna(row.ft_home) or pd.isna(row.ft_away):
            return None
        actual = 1 if row.ft_home > row.ft_away else 0 if row.ft_home == row.ft_away else 2
        return actual == {"MS 1": 1, "MS X": 0, "MS 2": 2}[label]
    target = TARGETS[label]
    if pd.isna(row[target]):
        return None
    return bool(row[target] == 1)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Run convert_excel_to_json.py first; missing {INPUT}")
    data = pd.read_json(INPUT, convert_dates=["date"])
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"]).copy()
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    predictions = []
    for day in sorted(data.date.unique()):
        train = data[(data.date < day) & data.target_o25.notna()]
        fixtures = data[data.date == day]
        if len(train) < 30:
            continue
        fitted = {label: fit_ensemble(train, target) for label, target in TARGETS.items()}
        fitted["MS"] = fit_ensemble(train, "target_ms", multiclass=True)

        fixture_rows = fixtures.reset_index(drop=True)
        probability_tables = {name: average_probabilities(models, fixture_rows[FEATURES])
                              for name, models in fitted.items()}
        for position, row in fixture_rows.iterrows():
            x = fixture_rows.loc[[position], FEATURES]
            if x.notna().sum(axis=1).iloc[0] < 3:
                continue
            market_probs = {name: {cls: float(values[position]) for cls, values in probs.items()}
                            for name, probs in probability_tables.items()}
            options = []
            ms_probs = market_probs["MS"]
            if ms_probs:
                ms_class = max(ms_probs, key=ms_probs.get)
                options.append({"label": MS_LABELS[int(ms_class)], "probability": ms_probs[ms_class]})
            for name in TARGETS:
                probs = market_probs[name]
                if probs:
                    options.append({"label": name, "probability": probs.get(1, 0.0)})
            if not options:
                continue
            options.sort(key=lambda item: item["probability"], reverse=True)
            primary = options[0]
            result = outcome_result(row, primary["label"])
            score = ""
            if pd.notna(row.ft_home) and pd.notna(row.ft_away):
                score = f"{int(row.ft_home)} - {int(row.ft_away)}"
            ht_score = ""
            if pd.notna(row.ht_home) and pd.notna(row.ht_away):
                ht_score = f"{int(row.ht_home)} - {int(row.ht_away)}"
            predictions.append({
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "time": str(row.time), "league": str(row.league),
                "home": str(row.home), "away": str(row.away),
                "primary": primary,
                "others": options[1:],
                "primary_correct": result,
                "score": score,
                "ht_score": ht_score,
            })
        print(f"{pd.Timestamp(day).date()}: {len(fixtures)} fixtures, {len(train)} earlier completed matches")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"predictions": predictions}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(predictions):,} predictions to {OUTPUT} ({OUTPUT.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
