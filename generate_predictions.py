
"""Incremental, chronological and leakage-safe ensemble predictions.

- Score updates do not trigger model retraining.
- Existing predictions are preserved.
- New dates are trained and predicted only once.
- Private model probabilities are stored separately from public predictions.
- Performance ledger is rebuilt from cached probabilities and current results.
"""

from __future__ import annotations

from pathlib import Path
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
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

STATE_DIR = ROOT / "cache"
STATE_FILE = STATE_DIR / "predictions_state.json"
TEMP_STATE = STATE_DIR / "predictions_state.tmp.json"

PROBS_FILE = STATE_DIR / "model_probabilities.json"
TEMP_PROBS = STATE_DIR / "model_probabilities.tmp.json"

STATE_VERSION = 2

FEATURES = [f"odds_{i}" for i in range(36, 75)]

MODEL_NAMES = ("rf", "xgb", "lgbm", "catboost")
SOURCES = MODEL_NAMES + ("odds",)

MARKETS = {
    "MS": {
        "target": "target_ms",
        "odds": {1: "odds_36", 0: "odds_37", 2: "odds_38"},
        "multiclass": True,
    },
    "2,5 Üst": {
        "target": "target_o25",
        "odds": {1: "odds_52"},
        "multiclass": False,
    },
    "İY 1,5 Üst": {
        "target": "target_ht15",
        "odds": {1: "odds_65"},
        "multiclass": False,
    },
    "KG Var": {
        "target": "target_btts",
        "odds": {1: "odds_55"},
        "multiclass": False,
    },
}

MS_LABELS = {1: "MS 1", 0: "MS X", 2: "MS 2"}

MIN_TRAIN_ROWS = 30
MIN_EXACT_ODDS_SAMPLE = 5
MIN_WEIGHT_SAMPLE = 30
LEAGUE_WEIGHT_SAMPLE = 30
BRIER_TEMPERATURE = 0.12

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

    parts = [
        day,
        row.get("time", ""),
        row.get("league", ""),
        row.get("home", ""),
        row.get("away", ""),
    ]

    return "|".join(
        " ".join(str(value or "").strip().casefold().split())
        for value in parts
    )


def canonical_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)


def input_fingerprint(data: pd.DataFrame) -> str:
    """
    Only fixture identity and opening odds are checked.

    Scores and target columns are deliberately excluded.
    Therefore ordinary score updates do not invalidate the cache.
    """

    columns = [
        "date",
        "time",
        "league",
        "home",
        "away",
        *FEATURES,
    ]

    columns = [
        column for column in columns
        if column in data.columns
    ]

    digest = hashlib.sha256()

    for row in data[columns].itertuples(
        index=False,
        name=None,
    ):
        values = [
            canonical_value(value)
            for value in row
        ]

        encoded = json.dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        digest.update(encoded)
        digest.update(b"\n")

    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def fit_models(
    train: pd.DataFrame,
    target: str,
    multiclass: bool = False,
):
    sample = train.dropna(subset=[target])

    if (
        len(sample) < MIN_TRAIN_ROWS
        or sample[target].nunique() < 2
    ):
        return {}

    classes = sorted(
        sample[target].astype(int).unique().tolist()
    )

    encoded = sample[target].astype(int).map(
        {
            value: index
            for index, value in enumerate(classes)
        }
    )

    X = sample[FEATURES]
    n_classes = len(classes)

    xgb_args = dict(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2,
        objective=(
            "multi:softprob"
            if multiclass
            else "binary:logistic"
        ),
        eval_metric=(
            "mlogloss"
            if multiclass
            else "logloss"
        ),
        n_jobs=2,
        random_state=42,
    )

    lgb_args = dict(
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=31,
        objective=(
            "multiclass"
            if multiclass
            else "binary"
        ),
        class_weight="balanced",
        verbosity=-1,
        n_jobs=2,
        random_state=42,
    )

    if multiclass:
        xgb_args["num_class"] = n_classes
        lgb_args["num_class"] = n_classes

    estimators = {
        "rf": RandomForestClassifier(
            n_estimators=180,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=42,
        ),

        "xgb": XGBClassifier(**xgb_args),

        "lgbm": LGBMClassifier(**lgb_args),

        "catboost": CatBoostClassifier(
            iterations=180,
            depth=6,
            learning_rate=0.045,
            verbose=False,
            loss_function=(
                "MultiClass"
                if multiclass
                else "Logloss"
            ),
            auto_class_weights="Balanced",
            random_seed=42,
            thread_count=2,
        ),
    }

    fitted = {}

    for name, estimator in estimators.items():
        try:
            pipeline = make_pipeline(
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
                estimator,
            )

            pipeline.fit(X, encoded)

            fitted[name] = {
                "model": pipeline,
                "classes": classes,
            }

        except Exception as exc:
            log(
                f"  {target}/{name} modeli atlandı: {exc}"
            )

    return fitted


def predict_model_probabilities(
    fitted,
    X,
    multiclass: bool,
):
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
                result[name] = (
                    float(raw[:, classes.index(1)][0])
                    if 1 in classes
                    else None
                )

        except Exception as exc:
            log(
                f"  Tahmin uyarısı ({name}): {exc}"
            )

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
    index = {}

    for market, config in MARKETS.items():
        target = config["target"]

        for outcome, odds_col in config["odds"].items():
            if odds_col not in data.columns:
                continue

            subset = data[
                ["date", odds_col, target]
            ].dropna(
                subset=["date", odds_col, target]
            )

            buckets = defaultdict(list)

            for (
                date_value,
                odd_value,
                target_value,
            ) in subset.itertuples(
                index=False,
                name=None,
            ):
                odd = safe_float(odd_value)

                if odd is None or odd <= 1.0:
                    continue

                hit = int(
                    int(target_value) == int(outcome)
                )

                buckets[round(odd, 2)].append(
                    (
                        pd.Timestamp(date_value).value,
                        hit,
                    )
                )

            for odd, values in buckets.items():
                values.sort(
                    key=lambda item: item[0]
                )

                index[(market, int(outcome), odd)] = {
                    "dates": [
                        item[0] for item in values
                    ],
                    "prefix": np.cumsum(
                        [item[1] for item in values],
                        dtype=np.int64,
                    ).tolist(),
                }

    return index


def exact_odds_probability(
    odds_index,
    market: str,
    outcome: int,
    odd_value,
    day_ns: int,
):
    odd = safe_float(odd_value)

    if odd is None or odd <= 1.0:
        return None, 0

    entry = odds_index.get(
        (
            market,
            int(outcome),
            round(odd, 2),
        )
    )

    if not entry:
        return None, 0

    count = bisect_left(
        entry["dates"],
        day_ns,
    )

    if count < MIN_EXACT_ODDS_SAMPLE:
        return None, count

    wins = entry["prefix"][count - 1]

    return (
        float((wins + 0.5) / (count + 1.0)),
        count,
    )


def build_odds_probabilities(
    row,
    odds_index,
    day_ns,
):
    odds_source = {}
    odds_samples = {}

    for market, config in MARKETS.items():
        odds_probs = {}
        counts = {}

        for outcome, odds_col in config["odds"].items():
            probability, count = exact_odds_probability(
                odds_index,
                market,
                outcome,
                row.get(odds_col),
                day_ns,
            )

            counts[str(outcome)] = count

            if probability is not None:
                odds_probs[int(outcome)] = probability

        odds_samples[market] = counts

        if market == "MS":
            if len(odds_probs) == 3:
                odds_source[market] = normalize_distribution(
                    odds_probs
                )

        elif 1 in odds_probs:
            odds_source[market] = float(
                odds_probs[1]
            )

    return odds_source, odds_samples


def brier_score(
    market: str,
    probabilities,
    actual,
):
    if actual is None or probabilities is None:
        return None

    if market == "MS":
        if not isinstance(probabilities, dict):
            return None

        return sum(
            (
                float(
                    probabilities.get(
                        cls,
                        probabilities.get(str(cls), 0.0),
                    )
                )
                - (
                    1.0
                    if cls == actual
                    else 0.0
                )
            ) ** 2
            for cls in (1, 0, 2)
        ) / 3.0

    try:
        return (
            float(probabilities)
            - float(actual)
        ) ** 2

    except (TypeError, ValueError):
        return None


class PerformanceLedger:
    def __init__(self):
        self.global_stats = defaultdict(
            lambda: {
                source: [0.0, 0]
                for source in SOURCES
            }
        )

        self.league_stats = defaultdict(
            lambda: {
                source: [0.0, 0]
                for source in SOURCES
            }
        )

    def weights(
        self,
        league: str,
        market: str,
        available_sources,
    ):
        global_row = self.global_stats[market]

        league_row = self.league_stats[
            (str(league), market)
        ]

        league_n = max(
            (
                league_row[source][1]
                for source in available_sources
            ),
            default=0,
        )

        use_league = (
            league_n >= LEAGUE_WEIGHT_SAMPLE
        )

        stats = (
            league_row
            if use_league
            else global_row
        )

        sample_n = max(
            (
                stats[source][1]
                for source in available_sources
            ),
            default=0,
        )

        if sample_n < MIN_WEIGHT_SAMPLE:
            weights = {
                source: 1.0 / len(available_sources)
                for source in available_sources
            }

            scope = "eşit ağırlık (yeterli geçmiş yok)"

        else:
            raw = {}

            for source in available_sources:
                total, count = stats[source]

                if count:
                    raw[source] = math.exp(
                        -(total / count)
                        / BRIER_TEMPERATURE
                    )

                else:
                    raw[source] = 1.0

            denominator = sum(raw.values()) or 1.0

            weights = {
                source: value / denominator
                for source, value in raw.items()
            }

            scope = (
                "lig"
                if use_league
                else "genel market"
            )

        return weights, scope, sample_n

    def add_day(self, day_records):
        for record in day_records:
            league = str(
                record.get("league", "")
            )

            source_probs = record.get(
                "source_probs",
                {},
            )

            for market, actual in record.get(
                "actuals",
                {},
            ).items():

                for source in SOURCES:
                    score = brier_score(
                        market,
                        source_probs.get(
                            source,
                            {},
                        ).get(market),
                        actual,
                    )

                    if score is None:
                        continue

                    self.global_stats[market][source][0] += score
                    self.global_stats[market][source][1] += 1

                    self.league_stats[
                        (league, market)
                    ][source][0] += score

                    self.league_stats[
                        (league, market)
                    ][source][1] += 1

    def to_dict(self):
        leagues = []

        for (league, market), stats in (
            self.league_stats.items()
        ):
            leagues.append({
                "league": league,
                "market": market,
                "stats": stats,
            })

        return {
            "global_stats": dict(
                self.global_stats
            ),
            "league_stats": leagues,
        }


def normalize_distribution(values):
    clean = {
        int(key): max(0.0, float(value))
        for key, value in values.items()
        if value is not None
    }

    total = sum(clean.values())

    if total <= 0:
        return {}

    return {
        key: value / total
        for key, value in clean.items()
    }


def normalize_source_probs(source_probs):
    """Restore integer MS class keys after JSON loading."""

    restored = {
        source: {}
        for source in SOURCES
    }

    for source, markets in source_probs.items():
        if source not in restored:
            continue

        for market, probabilities in markets.items():
            if market == "MS" and isinstance(
                probabilities,
                dict,
            ):
                restored[source][market] = {
                    int(key): float(value)
                    for key, value in probabilities.items()
                }

            elif probabilities is not None:
                restored[source][market] = float(
                    probabilities
                )

    return restored


def combine_market_sources(
    market: str,
    source_probs: dict,
    weights: dict,
):
    available = [
        source
        for source in weights
        if source_probs.get(
            source,
            {},
        ).get(market) is not None
    ]

    if not available:
        return {}, {}

    used = {
        source: weights[source]
        for source in available
    }

    total_weight = sum(used.values()) or 1.0

    used = {
        source: weight / total_weight
        for source, weight in used.items()
    }

    if market == "MS":
        combined = {}

        for cls in (1, 0, 2):
            combined[cls] = sum(
                used[source]
                * float(
                    source_probs[source][market].get(
                        cls,
                        0.0,
                    )
                )
                for source in available
            )

        return (
            normalize_distribution(combined),
            used,
        )

    probability = sum(
        used[source]
        * float(
            source_probs[source][market]
        )
        for source in available
    )

    return (
        {
            1: float(
                np.clip(probability, 0.0, 1.0)
            )
        },
        used,
    )


def build_source_probabilities(
    row,
    fitted_by_market,
    odds_index,
    day_ns,
):
    one_row = pd.DataFrame([row])[FEATURES]

    source_probs = {
        source: {}
        for source in SOURCES
    }

    for market, config in MARKETS.items():
        fitted = fitted_by_market.get(
            market,
            {},
        )

        model_probs = predict_model_probabilities(
            fitted,
            one_row,
            config["multiclass"],
        )

        for source, probabilities in model_probs.items():
            source_probs[source][market] = probabilities

    odds_source, odds_samples = build_odds_probabilities(
        row,
        odds_index,
        day_ns,
    )

    source_probs["odds"] = odds_source

    return source_probs, odds_samples


def make_joint_prediction(
    row,
    source_probs,
    ledger,
    odds_samples,
):
    league = str(row.get("league", ""))

    joint = {}
    source_weights = {}
    weight_scopes = {}

    for market in MARKETS:
        available = [
            source
            for source in SOURCES
            if source_probs.get(
                source,
                {},
            ).get(market) is not None
        ]

        if not available:
            continue

        weights, scope, sample_n = ledger.weights(
            league,
            market,
            available,
        )

        probabilities, used_weights = combine_market_sources(
            market,
            source_probs,
            weights,
        )

        if not probabilities:
            continue

        source_weights[market] = {
            key: round(value, 5)
            for key, value in used_weights.items()
        }

        weight_scopes[market] = {
            "scope": scope,
            "sample": int(sample_n),
        }

        if market == "MS":
            for cls, label in MS_LABELS.items():
                if cls in probabilities:
                    joint[label] = {
                        "probability": float(
                            probabilities[cls]
                        ),
                        "correct": None,
                        "odds_samples": int(
                            odds_samples.get(
                                "MS",
                                {},
                            ).get(str(cls), 0)
                        ),
                    }

        else:
            joint[market] = {
                "probability": float(
                    probabilities.get(1, 0.0)
                ),
                "correct": None,
                "odds_samples": int(
                    odds_samples.get(
                        market,
                        {},
                    ).get("1", 0)
                ),
            }

    return (
        joint,
        source_weights,
        weight_scopes,
    )


def result_for_label(row, label):
    if label in MS_LABELS.values():
        actual = get_actual(row, "MS")

        if actual is None:
            return None

        wanted = {
            value: key
            for key, value in MS_LABELS.items()
        }[label]

        return actual == wanted

    if label not in MARKETS:
        return None

    actual = get_actual(row, label)

    return (
        None
        if actual is None
        else actual == 1
    )


def update_result_fields(prediction, row):
    prediction["time"] = str(
        row.get("time", "")
    )

    prediction["league"] = str(
        row.get("league", "")
    )

    prediction["home"] = str(
        row.get("home", "")
    )

    prediction["away"] = str(
        row.get("away", "")
    )

    prediction["score"] = ""
    prediction["ht_score"] = ""

    if (
        pd.notna(row.get("ft_home"))
        and pd.notna(row.get("ft_away"))
    ):
        prediction["score"] = (
            f"{int(row['ft_home'])} - "
            f"{int(row['ft_away'])}"
        )

    if (
        pd.notna(row.get("ht_home"))
        and pd.notna(row.get("ht_away"))
    ):
        prediction["ht_score"] = (
            f"{int(row['ht_home'])} - "
            f"{int(row['ht_away'])}"
        )

    for label, item in prediction.get(
        "joint_markets",
        {},
    ).items():
        item["correct"] = result_for_label(
            row,
            label,
        )

    primary_label = prediction.get(
        "primary",
        {},
    ).get("label", "")

    prediction["primary_correct"] = result_for_label(
        row,
        primary_label,
    )

    prediction["market_results"] = {
        market: get_actual(row, market)
        for market in MARKETS
    }

    return prediction


def load_input():
    if not INPUT.exists():
        raise FileNotFoundError(
            "Önce convert_excel_to_json.py çalıştırılmalı: "
            f"{INPUT}"
        )

    data = pd.read_json(
        INPUT,
        convert_dates=["date"],
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    data = data.dropna(
        subset=["date"]
    ).copy()

    for column in FEATURES:
        if column not in data.columns:
            data[column] = np.nan

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    for config in MARKETS.values():
        target = config["target"]

        if target not in data.columns:
            data[target] = np.nan

        data[target] = pd.to_numeric(
            data[target],
            errors="coerce",
        )

    for column in (
        "ft_home",
        "ft_away",
        "ht_home",
        "ht_away",
    ):
        if column not in data.columns:
            data[column] = np.nan

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    for column in (
        "time",
        "league",
        "home",
        "away",
    ):
        if column not in data.columns:
            data[column] = ""

        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
        )

    data = data.sort_values(
        [
            "date",
            "time",
            "league",
            "home",
            "away",
        ],
        kind="stable",
    ).reset_index(drop=True)

    # Mevcut veri filtresi aynen korunuyor.
    valid_odds = (
        data[FEATURES]
        .notna()
        .sum(axis=1)
        >= 3
    )

    data = data.loc[
        valid_odds
    ].copy()

    if data.empty:
        raise RuntimeError(
            "En az üç açılış oranı bulunan maç bulunamadı."
        )

    return data


def load_existing_output():
    if not OUTPUT.exists():
        return []

    try:
        payload = json.loads(
            OUTPUT.read_text(
                encoding="utf-8"
            )
        )

        predictions = payload.get(
            "predictions",
            [],
        )

        if not isinstance(predictions, list):
            return []

        return predictions

    except Exception:
        return []


def load_probability_cache():
    if not PROBS_FILE.exists():
        return None

    try:
        payload = json.loads(
            PROBS_FILE.read_text(
                encoding="utf-8"
            )
        )

        raw = payload.get(
            "fixture_probs",
            {},
        )

        return {
            key: normalize_source_probs(value)
            for key, value in raw.items()
        }

    except Exception as exc:
        log(
            "Model olasılık cache okunamadı: "
            f"{exc}"
        )

        return None


def load_incremental_state(data):
    """
    Cache is valid only if:
    - State version matches.
    - Public predictions file is intact.
    - Private probability cache is intact.
    - Fixture identities and opening odds up to last date match.

    Scores and targets are not part of the fingerprint.
    """

    if (
        not STATE_FILE.exists()
        or not OUTPUT.exists()
        or not PROBS_FILE.exists()
    ):
        return None

    try:
        state = json.loads(
            STATE_FILE.read_text(
                encoding="utf-8"
            )
        )

        if state.get("version") != STATE_VERSION:
            return None

        if (
            state.get("output_sha256")
            != file_sha256(OUTPUT)
        ):
            return None

        if (
            state.get("probabilities_sha256")
            != file_sha256(PROBS_FILE)
        ):
            return None

        last_date = state.get(
            "last_processed_date"
        )

        if not last_date:
            return None

        prefix = data[
            data["date"] <= pd.Timestamp(last_date)
        ].copy()

        current_fingerprint = input_fingerprint(
            prefix
        )

        if (
            current_fingerprint
            != state.get("input_fingerprint")
        ):
            log(
                "Geçmiş maç veya açılış oranı değişmiş."
            )

            return None

        predictions = load_existing_output()
        fixture_probs = load_probability_cache()

        if fixture_probs is None:
            return None

        return {
            "last_date": last_date,
            "predictions": predictions,
            "fixture_probs": fixture_probs,
        }

    except Exception as exc:
        log(
            "Önbellek okunamadı: "
            f"{exc}"
        )

        return None


def refresh_existing_predictions(
    data,
    predictions,
    fixture_probs,
    odds_index,
):
    """
    Update old match results without fitting any models.

    Also refresh historical exact-odds probabilities so corrected
    earlier results can affect later odds-history calculations.
    """

    row_lookup = {}

    for _, series in data.iterrows():
        row = series.to_dict()
        row_lookup[fixture_key(row)] = row

    refreshed = 0
    missing = 0

    for prediction in predictions:
        key = prediction.get(
            "fixture_id",
            "",
        )

        row = row_lookup.get(key)

        if row is None:
            missing += 1
            continue

        update_result_fields(
            prediction,
            row,
        )

        # Exact odds probabilities are recalculated using current results.
        day_ns = pd.Timestamp(
            row["date"]
        ).value

        odds_source, odds_samples = build_odds_probabilities(
            row,
            odds_index,
            day_ns,
        )

        if key in fixture_probs:
            fixture_probs[key]["odds"] = odds_source

        prediction["odds_samples"] = odds_samples

        refreshed += 1

    log(
        f"Eski tahmin sonuçları güncellendi: {refreshed:,}"
    )

    if missing:
        log(
            f"Veride bulunamayan eski tahmin: {missing:,}"
        )

    return row_lookup


def rebuild_ledger(
    predictions,
    fixture_probs,
    row_lookup,
):
    """
    Rebuild performance statistics from cached model probabilities
    and current actual results. No model training occurs here.
    """

    ledger = PerformanceLedger()

    grouped = defaultdict(list)

    for prediction in predictions:
        day = prediction.get("date", "")

        if day:
            grouped[day].append(prediction)

    for day in sorted(grouped):
        day_records = []

        for prediction in grouped[day]:
            key = prediction.get(
                "fixture_id",
                "",
            )

            source_probs = fixture_probs.get(key)

            row = row_lookup.get(key)

            if source_probs is None or row is None:
                continue

            actuals = {
                market: get_actual(row, market)
                for market in MARKETS
            }

            day_records.append({
                "league": str(
                    row.get("league", "")
                ),
                "source_probs": source_probs,
                "actuals": actuals,
            })

        # Same-day results are added only after that day's predictions.
        ledger.add_day(day_records)

    return ledger


def save_json_atomic(
    path: Path,
    temp_path: Path,
    payload,
):
    temp_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    os.replace(
        temp_path,
        path,
    )


def main():
    data = load_input()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT.exists() and not BACKUP.exists():
        shutil.copy2(
            OUTPUT,
            BACKUP,
        )

        log(
            f"Mevcut tahmin dosyası yedeklendi: {BACKUP}"
        )

    log(
        f"Toplam maç: {len(data):,}"
    )

    cached = load_incremental_state(data)

    if cached:
        predictions = cached["predictions"]
        fixture_probs = cached["fixture_probs"]
        last_processed_date = cached["last_date"]

        log(
            "Önbellek geçerli. Son tamamlanan tarih: "
            f"{last_processed_date}"
        )

        log(
            f"Mevcut tahmin korunuyor: {len(predictions):,}"
        )

    else:
        predictions = []
        fixture_probs = {}
        last_processed_date = None

        log(
            "Yeni cache sürümü veya geçmiş veri değişikliği."
        )

        log(
            "Bir defalık tam kronolojik hesaplama yapılacak."
        )

    log(
        "Exact oran geçmiş indeksi hazırlanıyor..."
    )

    odds_index = build_odds_index(data)

    # -------------------------------------------------
    # 1. EXISTING RESULTS REFRESH
    # -------------------------------------------------

    if predictions and fixture_probs:
        row_lookup = refresh_existing_predictions(
            data,
            predictions,
            fixture_probs,
            odds_index,
        )

        log(
            "Eski tahminlerin model olasılıkları korunuyor."
        )

        log(
            "Güncel sonuçlarla başarı istatistikleri hesaplanıyor."
        )

        ledger = rebuild_ledger(
            predictions,
            fixture_probs,
            row_lookup,
        )

    else:
        row_lookup = {
            fixture_key(series.to_dict()): series.to_dict()
            for _, series in data.iterrows()
        }

        ledger = PerformanceLedger()

    # -------------------------------------------------
    # 2. FIND NEW DATES
    # -------------------------------------------------

    unique_days = sorted(
        pd.Timestamp(value)
        for value in data["date"].drop_duplicates()
    )

    if last_processed_date:
        start_date = pd.Timestamp(
            last_processed_date
        )

        pending_days = [
            day
            for day in unique_days
            if day > start_date
        ]

    else:
        pending_days = unique_days

    log(
        f"Toplam tarih: {len(unique_days):,}"
    )

    log(
        f"Yeni işlenecek tarih: {len(pending_days):,}"
    )

    # -------------------------------------------------
    # 3. TRAIN ONLY FOR NEW DATES
    # -------------------------------------------------

    for day_number, day in enumerate(
        pending_days,
        start=1,
    ):
        day_rows = data[
            data["date"] == day
        ]

        train = data[
            data["date"] < day
        ]

        if len(train) < MIN_TRAIN_ROWS:
            log(
                f"{day.date()}: yeterli geçmiş eğitim verisi yok "
                f"({len(train)}), atlandı"
            )

            last_processed_date = day.strftime(
                "%Y-%m-%d"
            )

            continue

        log(
            f"[{day_number}/{len(pending_days)}] "
            f"{day.date()} modeller eğitiliyor..."
        )

        fitted_by_market = {}

        for market, config in MARKETS.items():
            fitted_by_market[market] = fit_models(
                train,
                config["target"],
                multiclass=config["multiclass"],
            )

        day_records_for_ledger = []

        day_ns = pd.Timestamp(day).value

        day_prediction_count = 0

        for _, series in day_rows.iterrows():
            row = series.to_dict()

            source_probs, odds_samples = (
                build_source_probabilities(
                    row,
                    fitted_by_market,
                    odds_index,
                    day_ns,
                )
            )

            joint, source_weights, weight_scopes = (
                make_joint_prediction(
                    row,
                    source_probs,
                    ledger,
                    odds_samples,
                )
            )

            if not joint:
                continue

            options = [
                {
                    "label": label,
                    "probability": item["probability"],
                }
                for label, item in joint.items()
            ]

            options.sort(
                key=lambda item: item["probability"],
                reverse=True,
            )

            key = fixture_key(row)

            prediction = {
                "date": pd.Timestamp(
                    day
                ).strftime("%Y-%m-%d"),

                "time": str(
                    row.get("time", "")
                ),

                "league": str(
                    row.get("league", "")
                ),

                "home": str(
                    row.get("home", "")
                ),

                "away": str(
                    row.get("away", "")
                ),

                "primary": options[0],
                "others": options[1:],
                "joint_markets": joint,
                "source_weights": source_weights,
                "weight_scopes": weight_scopes,
                "odds_samples": odds_samples,
                "primary_correct": None,
                "market_results": {},
                "score": "",
                "ht_score": "",
                "fixture_id": key,
            }

            prediction = update_result_fields(
                prediction,
                row,
            )

            predictions.append(prediction)

            # Private cache: not included in public predictions.json.
            fixture_probs[key] = source_probs

            day_prediction_count += 1

            actuals = {
                market: get_actual(row, market)
                for market in MARKETS
            }

            day_records_for_ledger.append({
                "league": str(
                    row.get("league", "")
                ),
                "source_probs": source_probs,
                "actuals": actuals,
            })

        # Same-day predictions are completed before ledger update.
        ledger.add_day(
            day_records_for_ledger
        )

        last_processed_date = day.strftime(
            "%Y-%m-%d"
        )

        log(
            f"{day.date()}: {day_prediction_count} tahmin; "
            f"eğitim maç sayısı={len(train):,}"
        )

    # -------------------------------------------------
    # 4. SORT AND SAVE PUBLIC PREDICTIONS
    # -------------------------------------------------

    predictions.sort(
        key=lambda item: (
            item.get("date", ""),
            item.get("time", ""),
            item.get("home", ""),
            item.get("away", ""),
        )
    )

    payload = {
        "generated_at": datetime.now(
            ZoneInfo("Europe/Istanbul")
        ).isoformat(),

        "method": (
            "walk_forward_5_source_league_market_brier"
        ),

        "markets": list(
            MARKETS.keys()
        ),

        "prediction_count": len(
            predictions
        ),

        "predictions": predictions,
    }

    save_json_atomic(
        OUTPUT,
        TEMP_OUTPUT,
        payload,
    )

    # -------------------------------------------------
    # 5. SAVE PRIVATE MODEL PROBABILITIES
    # -------------------------------------------------

    probabilities_payload = {
        "version": STATE_VERSION,
        "fixture_probs": fixture_probs,
    }

    save_json_atomic(
        PROBS_FILE,
        TEMP_PROBS,
        probabilities_payload,
    )

    # -------------------------------------------------
    # 6. SAVE STATE LAST
    # -------------------------------------------------

    if last_processed_date:
        completed_prefix = data[
            data["date"] <= pd.Timestamp(
                last_processed_date
            )
        ]

        state_payload = {
            "version": STATE_VERSION,

            "last_processed_date": (
                last_processed_date
            ),

            "input_fingerprint": input_fingerprint(
                completed_prefix
            ),

            "output_sha256": file_sha256(
                OUTPUT
            ),

            "probabilities_sha256": file_sha256(
                PROBS_FILE
            ),
        }

        save_json_atomic(
            STATE_FILE,
            TEMP_STATE,
            state_payload,
        )

    log(
        "Tamamlandı: "
        f"{len(predictions):,} tahmin -> "
        f"{OUTPUT} "
        f"({OUTPUT.stat().st_size:,} bayt)"
    )

    log(
        f"Özel model cache: {PROBS_FILE} "
        f"({PROBS_FILE.stat().st_size:,} bayt)"
    )

    log(
        f"Durum dosyası: {STATE_FILE}"
    )

    if BACKUP.exists():
        log(
            f"Yedek dosya: {BACKUP}"
        )


if __name__ == "__main__":
    main()
