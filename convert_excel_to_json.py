from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "input" / "Excel-Açılış-Cep-Telefonu.xlsx"
JSON = ROOT / "input" / "matches.json"
COLS = {
    "ht_home": 22, "ht_away": 23, "ft_home": 24, "ft_away": 25,
    "date": 30, "league": 32, "time": 33, "home": 34, "away": 35,
}
FEATURES = [f"odds_{i}" for i in range(36, 75)]


def parse_odds(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(" ", "").replace(",", "."))
    except ValueError:
        return np.nan


def main():
    if not XLSX.exists():
        raise FileNotFoundError(f"Input workbook not found: {XLSX}")
    raw = pd.read_excel(XLSX, header=None, sheet_name=0, engine="openpyxl")
    if raw.shape[1] < 75:
        raise ValueError("Expected workbook with at least 75 columns.")
    data = pd.DataFrame({name: raw.iloc[8:, index].reset_index(drop=True) for name, index in COLS.items()})
    for index in range(36, 75):
        data[f"odds_{index}"] = raw.iloc[8:, index].reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in FEATURES:
        data[column] = data[column].map(parse_odds)
    for column in ["ht_home", "ht_away", "ft_home", "ft_away"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["home", "away", "league", "time"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data = data.dropna(subset=["date"]).copy()
    data["target_ms"] = np.select(
        [data.ft_home > data.ft_away, data.ft_home == data.ft_away, data.ft_home < data.ft_away],
        [1, 0, 2], default=np.nan,
    )
    data["target_o25"] = np.where(
        data.ft_home.notna() & data.ft_away.notna(), ((data.ft_home + data.ft_away) > 2.5).astype(float), np.nan
    )
    data["target_ht15"] = np.where(
        data.ht_home.notna() & data.ht_away.notna(), ((data.ht_home + data.ht_away) > 1.5).astype(float), np.nan
    )
    data["target_btts"] = np.where(
        data.ft_home.notna() & data.ft_away.notna(), ((data.ft_home > 0) & (data.ft_away > 0)).astype(float), np.nan
    )
    data.to_json(JSON, orient="records", date_format="iso", force_ascii=False)
    print(f"Converted {len(data):,} matches to {JSON.relative_to(ROOT)} ({JSON.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
