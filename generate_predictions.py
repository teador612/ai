"""Chronological, leakage-safe ensemble predictions for the GitHub Pages site.

Every fixture is predicted using only matches dated strictly before its date.
The four existing ML models are retained. An exact-opening-odds historical
source is added, and source weights are learned from earlier out-of-sample
predictions using Brier scores, separately by market and (when sufficiently
large) league.
"""
from __future__ import annotations

from pathlib import Path
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import math
import os
import shutil
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
BACKUP = ROOT / "docs" / "predictions.backup.json"
TEMP_OUTPUT = ROOT / "docs" / "predictions.tmp.json"

FEATURES = [f"odds_{i}" for i in range(36, 75)]
MODEL_NAMES = ("rf", "xgb", "lgbm", "catboost")
SOURCES = MODEL_NAMES + ("odds",)

# Internal market keys and their source columns.
MARKETS = {
    "MS": {"target": "target_ms", "odds": {1: "odds_36", 0: "odds_37", 2: "odds_38"}, "multiclass": True},
    "2,5 Üst": {"target": "target_o25", "odds": {1: "odds_52"}, "multiclass": False},
    "İY 1,5 Üst": {"target": "target_ht15", "odds": {1: "odds_65"}, "multiclass": False},
    "KG Var": {"target": "target_btts", "odds": {1: "odds_55"}, "multiclass": False},
}
MS_LABELS = {1: "MS 1", 0: "MS X", 2: "MS 2"}
MIN_TRAIN_ROWS = 30
MIN_EXACT_ODDS_SAMPLE = 5
MIN_WEIGHT_SAMPLE = 30
LEAGUE_WEIGHT_SAMPLE = 30
Brier_TEMPERATURE = 0.12
warnings.filterwarnings("ignore", category=UserWarning)


def log(message: str) -> None:
    print(message, flush=True)


def safe_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def fixture_key(row) -> str:
    day = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    parts = [day, row.get("time", ""), row.get("league", ""), row.get("home", ""), row.get("away", "")]
    return "|".join(" ".join(str(value or "").strip().casefold().split()) for value in parts)


def fit_models(train: pd.DataFrame, target: str, multiclass: bool = False):
    """Fit the four existing estimators and return individually addressable models."""
    sample = train.dropna(subset=[target])
    if len(sample) < MIN_TRAIN_ROWS or sample[target].nunique() < 2:
        return {}

    classes = sorted(sample[target].astype(int).unique().tolist())
    encoded = sample[target].astype(int).map({value: i for i, value in enumerate(classes)})
    X = sample[FEATURES]
    n_classes = len(classes)

    xgb_args = dict(
        n_estimators=180, max_depth=5, learning_rate=0.045,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=2,
        objective="multi:softprob" if multiclass else "binary:logistic",
        eval_metric="mlogloss" if multiclass else "logloss",
        n_jobs=2, random_state=42,
    )
    lgb_args = dict(
        n_estimators=180, learning_rate=0.045, num_leaves=31,
        objective="multiclass" if multiclass else "binary",
        class_weight="balanced", verbosity=-1, n_jobs=2, random_state=42,
    )
    if multiclass:
        xgb_args["num_class"] = n_classes
        lgb_args["num_class"] = n_classes

    estimators = {
        "rf": RandomForestClassifier(
            n_estimators=180, max_features="sqrt", min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=2, random_state=42,
        ),
        "xgb": XGBClassifier(**xgb_args),
        "lgbm": LGBMClassifier(**lgb_args),
        "catboost": CatBoostClassifier(
            iterations=180, depth=6, learning_rate=0.045, verbose=False,
            loss_function="MultiClass" if multiclass else "Logloss",
            auto_class_weights="Balanced", random_seed=42, thread_count=2,
        ),
    }

    fitted = {}
    for name, estimator in estimators.items():
        try:
            pipeline = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)
            pipeline.fit(X, encoded)
            fitted[name] = {"model": pipeline, "classes": classes}
        except Exception as exc:  # A failed estimator must not stop the other sources.
            log(f"  {target}/{name} modeli atlandı: {exc}")
    return fitted


def predict_model_probabilities(fitted: dict, X: pd.DataFrame, multiclass: bool):
    result = {}
    for name, item in fitted.items():
        try:
            raw = item["model"].predict_proba(X)
            classes = item["classes"]
            if multiclass:
                result[name] = {
                    int(label): float(raw[:, index][0])
                    for index, label in enumerate(classes)
                }
            else:
                result[name] = float(raw[:, classes.index(1)][0]) if 1 in classes else None
        except Exception as exc:
            log(f"  Tahmin uyarısı ({name}): {exc}")
    return result


def get_actual(row, market: str):
    target = MARKETS[market]["target"]
    value = row.get(target)
    if pd.isna(value):
        return None
    if market == "MS":
        return int(value)
    return int(float(value) == 1)


def build_odds_index(data: pd.DataFrame):
    """key -> sorted dates and outcomes, so each query can use only prior dates."""
    index = {}
    for market, config in MARKETS.items():
        target = config["target"]
        for outcome, odds_col in config["odds"].items():
            if odds_col not in data.columns:
                continue
            subset = data[["date", odds_col, target]].dropna(subset=["date", odds_col, target])
            buckets = defaultdict(list)
            for date_value, odd_value, target_value in subset.itertuples(index=False, name=None):
                odd = safe_float(odd_value)
                if odd is None or odd <= 1.0:
                    continue
                actual = int(target_value)
                hit = int(actual == int(outcome))
                buckets[round(odd, 2)].append((pd.Timestamp(date_value).value, hit))
            for odd, values in buckets.items():
                values.sort(key=lambda item: item[0])
                index[(market, int(outcome), odd)] = {
                    "dates": [item[0] for item in values],
                    "hits": [item[1] for item in values],
                    "prefix": np.cumsum([item[1] for item in values], dtype=np.int64).tolist(),
                }
    return index


def exact_odds_probability(odds_index, market: str, outcome: int, odd_value, day_ns: int):
    odd = safe_float(odd_value)
    if odd is None or odd <= 1.0:
        return None, 0
    entry = odds_index.get((market, int(outcome), round(odd, 2)))
    if not entry:
        return None, 0
    count = bisect_left(entry["dates"], day_ns)  # strict: date must be earlier
    if count < MIN_EXACT_ODDS_SAMPLE:
        return None, count
    wins = entry["prefix"][count - 1]
    # Jeffreys smoothing prevents 0%/100% estimates from small samples.
    return float((wins + 0.5) / (count + 1.0)), count


def brier_score(market: str, probabilities, actual):
    if actual is None or probabilities is None:
        return None
    if market == "MS":
        if not isinstance(probabilities, dict):
            return None
        # Three-class Brier score, normalized by number of classes.
        return sum((float(probabilities.get(cls, 0.0)) - (1.0 if cls == actual else 0.0)) ** 2
                   for cls in (1, 0, 2)) / 3.0
    try:
        return (float(probabilities) - float(actual)) ** 2
    except (TypeError, ValueError):
        return None


class PerformanceLedger:
    """Cumulative out-of-sample Brier scores; updates happen only after each day."""
    def __init__(self):
        self.global_stats = defaultdict(lambda: {source: [0.0, 0] for source in SOURCES})
        self.league_stats = defaultdict(lambda: {source: [0.0, 0] for source in SOURCES})

    def weights(self, league: str, market: str, available_sources):
        global_row = self.global_stats[market]
        league_row = self.league_stats[(str(league), market)]
        league_n = max((league_row[s][1] for s in available_sources), default=0)
        use_league = league_n >= LEAGUE_WEIGHT_SAMPLE
        stats = league_row if use_league else global_row
        sample_n = max((stats[s][1] for s in available_sources), default=0)

        if sample_n < MIN_WEIGHT_SAMPLE:
            weights = {source: 1.0 / len(available_sources) for source in available_sources}
            scope = "eşit ağırlık (yeterli geçmiş yok)"
        else:
            raw = {}
            for source in available_sources:
                total, count = stats[source]
                if count:
                    score = total / count
                    raw[source] = math.exp(-score / Brier_TEMPERATURE)
                else:
                    raw[source] = 1.0
            denominator = sum(raw.values()) or 1.0
            weights = {source: value / denominator for source, value in raw.items()}
            scope = "lig" if use_league else "genel market"
        return weights, scope, sample_n

    def add_day(self, day_records):
        # This is deliberately called after all fixtures on the same date are predicted.
        for record in day_records:
            league = str(record.get("league", ""))
            for market, actual in record.get("actuals", {}).items():
                source_probs = record.get("source_probs", {})
                for source in SOURCES:
                    score = brier_score(market, source_probs.get(source, {}).get(market), actual)
                    if score is None:
                        continue
                    self.global_stats[market][source][0] += score
                    self.global_stats[market][source][1] += 1
                    self.league_stats[(league, market)][source][0] += score
                    self.league_stats[(league, market)][source][1] += 1


def normalize_distribution(values):
    clean = {int(key): max(0.0, float(value)) for key, value in values.items() if value is not None}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in clean.items()}


def combine_market_sources(market: str, source_probs: dict, weights: dict):
    available = [source for source in weights if source_probs.get(source, {}).get(market) is not None]
    if not available:
        return {}, {}
    used = {source: weights[source] for source in available}
    total_weight = sum(used.values()) or 1.0
    used = {source: weight / total_weight for source, weight in used.items()}

    if market == "MS":
        classes = (1, 0, 2)
        combined = {}
        for cls in classes:
            combined[cls] = sum(
                used[source] * float(source_probs[source][market].get(cls, 0.0))
                for source in available
            )
        return normalize_distribution(combined), used

    probability = sum(used[source] * float(source_probs[source][market]) for source in available)
    return {1: float(np.clip(probability, 0.0, 1.0))}, used


def build_source_probabilities(row, fitted_by_market, odds_index, day_ns):
    one_row = pd.DataFrame([row])[FEATURES]
    source_probs = {source: {} for source in SOURCES}
    odds_samples = {}

    for market, config in MARKETS.items():
        fitted = fitted_by_market.get(market, {})
        model_probs = predict_model_probabilities(fitted, one_row, config["multiclass"])
        for source, probabilities in model_probs.items():
            source_probs[source][market] = probabilities

        odds_probs = {}
        counts = {}
        for outcome, odds_col in config["odds"].items():
            probability, count = exact_odds_probability(
                odds_index, market, outcome, row.get(odds_col), day_ns
            )
            counts[str(outcome)] = count
            if probability is not None:
                odds_probs[int(outcome)] = probability
        odds_samples[market] = counts

        if market == "MS":
            # Independent exact-odds hit rates are normalized to a 1/X/2 distribution.
            if len(odds_probs) == 3:
                source_probs["odds"][market] = normalize_distribution(odds_probs)
        elif 1 in odds_probs:
            source_probs["odds"][market] = float(odds_probs[1])

    return source_probs, odds_samples


def make_joint_prediction(row, source_probs, ledger, odds_samples):
    league = str(row.get("league", ""))
    joint = {}
    source_weights = {}
    weight_scopes = {}

    for market in MARKETS:
        available = [source for source in SOURCES if source_probs.get(source, {}).get(market) is not None]
        if not available:
            continue
        weights, scope, sample_n = ledger.weights(league, market, available)
        probabilities, used_weights = combine_market_sources(market, source_probs, weights)
        if not probabilities:
            continue
        source_weights[market] = {key: round(value, 5) for key, value in used_weights.items()}
        weight_scopes[market] = {"scope": scope, "sample": int(sample_n)}

        if market == "MS":
            for cls, label in MS_LABELS.items():
                if cls in probabilities:
                    joint[label] = {
                        "probability": float(probabilities[cls]),
                        "correct": None,
                        "odds_samples": int(odds_samples.get("MS", {}).get(str(cls), 0)),
                    }
        else:
            label = market
            joint[label] = {
                "probability": float(probabilities.get(1, 0.0)),
                "correct": None,
                "odds_samples": int(odds_samples.get(market, {}).get("1", 0)),
            }

    return joint, source_weights, weight_scopes


def result_for_label(row, label):
    if label in MS_LABELS.values():
        actual = get_actual(row, "MS")
        if actual is None:
            return None
        wanted = {value: key for key, value in MS_LABELS.items()}[label]
        return actual == wanted
    market = label if label in MARKETS else None
    if market is None:
        return None
    actual = get_actual(row, market)
    return None if actual is None else actual == 1


def update_result_fields(prediction, row):
    prediction["time"] = str(row.get("time", ""))
    prediction["league"] = str(row.get("league", ""))
    prediction["home"] = str(row.get("home", ""))
    prediction["away"] = str(row.get("away", ""))
    prediction["score"] = ""
    prediction["ht_score"] = ""
    if pd.notna(row.get("ft_home")) and pd.notna(row.get("ft_away")):
        prediction["score"] = f"{int(row['ft_home'])} - {int(row['ft_away'])}"
    if pd.notna(row.get("ht_home")) and pd.notna(row.get("ht_away")):
        prediction["ht_score"] = f"{int(row['ht_home'])} - {int(row['ht_away'])}"

    for label, item in prediction.get("joint_markets", {}).items():
        item["correct"] = result_for_label(row, label)

    primary_label = prediction.get("primary", {}).get("label", "")
    prediction["primary_correct"] = result_for_label(row, primary_label)
    prediction["market_results"] = {
        market: get_actual(row, market) for market in MARKETS
    }
    return prediction


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Önce convert_excel_to_json.py çalıştırılmalı: {INPUT}")

    data = pd.read_json(INPUT, convert_dates=["date"])
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"]).copy()
    for column in FEATURES:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for config in MARKETS.values():
        target = config["target"]
        if target not in data.columns:
            data[target] = np.nan
        data[target] = pd.to_numeric(data[target], errors="coerce")

    data = data.sort_values(["date", "time", "league", "home", "away"], kind="stable").reset_index(drop=True)
    valid_odds = data[FEATURES].notna().sum(axis=1) >= 3
    data = data.loc[valid_odds].copy()
    if data.empty:
        raise RuntimeError("En az üç açılış oranı bulunan maç bulunamadı.")

    # Preserve the original JSON before replacing it. Never overwrite an existing backup.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and not BACKUP.exists():
        shutil.copy2(OUTPUT, BACKUP)
        log(f"Mevcut tahmin dosyası yedeklendi: {BACKUP}")

    log(f"Toplam maç: {len(data):,}")
    log("Exact oran geçmiş indeksi hazırlanıyor...")
    odds_index = build_odds_index(data)
    ledger = PerformanceLedger()
    predictions = []
    unique_days = sorted(pd.Timestamp(value) for value in data["date"].drop_duplicates())
    log(f"Kronolojik tahmin günü: {len(unique_days):,}")

    for day_number, day in enumerate(unique_days, start=1):
        day_rows = data[data["date"] == day]
        train = data[data["date"] < day]
        if len(train) < MIN_TRAIN_ROWS:
            log(f"{day.date()}: yeterli geçmiş eğitim verisi yok ({len(train)}), atlandı")
            continue

        fitted_by_market = {}
        for market, config in MARKETS.items():
            fitted_by_market[market] = fit_models(
                train, config["target"], multiclass=config["multiclass"]
            )

        day_records_for_ledger = []
        day_ns = pd.Timestamp(day).value
        for _, series in day_rows.iterrows():
            row = series.to_dict()
            source_probs, odds_samples = build_source_probabilities(
                row, fitted_by_market, odds_index, day_ns
            )
            joint, source_weights, weight_scopes = make_joint_prediction(
                row, source_probs, ledger, odds_samples
            )
            if not joint:
                continue

            options = [
                {"label": label, "probability": item["probability"]}
                for label, item in joint.items()
            ]
            options.sort(key=lambda item: item["probability"], reverse=True)
            key = fixture_key(row)
            prediction = {
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "time": str(row.get("time", "")),
                "league": str(row.get("league", "")),
                "home": str(row.get("home", "")),
                "away": str(row.get("away", "")),
                "primary": options[0],
                "others": options[1:],
                "joint_markets": joint,
                "source_probabilities": source_probs,
                "source_weights": source_weights,
                "weight_scopes": weight_scopes,
                "odds_samples": odds_samples,
                "primary_correct": None,
                "market_results": {},
                "score": "",
                "ht_score": "",
                "fixture_id": key,
            }
            prediction = update_result_fields(prediction, row)
            predictions.append(prediction)

            actuals = {market: get_actual(row, market) for market in MARKETS}
            day_records_for_ledger.append({
                "league": str(row.get("league", "")),
                "source_probs": source_probs,
                "actuals": actuals,
            })

        # Update model performance only after every match of this date was predicted.
        ledger.add_day(day_records_for_ledger)
        log(
            f"[{day_number}/{len(unique_days)}] {day.date()}: "
            f"{len(day_records_for_ledger)} maç tahmin edildi; "
            f"önceki eğitim maçı={len(train):,}"
        )

    predictions.sort(key=lambda item: (item["date"], item["time"], item["home"], item["away"]))
    payload = {
        "generated_at": datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(),
        "method": "walk_forward_5_source_league_market_brier",
        "markets": list(MARKETS.keys()),
        "prediction_count": len(predictions),
        "predictions": predictions,
    }
    TEMP_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    # Atomic replace: a failed generation never leaves a partially written predictions.json.
    os.replace(TEMP_OUTPUT, OUTPUT)
    log(f"Tamamlandı: {len(predictions):,} tahmin -> {OUTPUT} ({OUTPUT.stat().st_size:,} bayt)")
    log(f"Yedek dosya: {BACKUP if BACKUP.exists() else 'önceden mevcut yedek yok'}")


if __name__ == "__main__":
    main()
