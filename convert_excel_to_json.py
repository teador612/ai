
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "input" / "Excel-Açılış-Cep-Telefonu.xlsx"
JSON = ROOT / "input" / "matches.json"

# Excel sütunları A'dan başladığı için sıfır tabanlı indeksler.
COLS = {
    "score_ht": 21,  # V: SKOR İY
    "score_ms": 22,  # W: SKOR MS
    "date": 23,      # X: TARİH
    "league": 25,    # Z: LİG
    "time": 26,      # AA: SAAT
    "home": 27,      # AB: EV SAHİBİ
    "away": 28,      # AC: DEPLASMAN
}

# Modelin mevcut özellik isimlerini koruyoruz.
# Excel'deki gerçek oran sütunları AD (29) - BP (67).
FEATURES = [f"odds_{i}" for i in range(36, 75)]
ODDS_START = 29


def parse_odds(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(
            str(value).strip().replace(" ", "").replace(",", ".")
        )
    except (ValueError, TypeError):
        return np.nan


def parse_score(value):
    """1-0, 1:0 veya 1 - 0 biçimindeki skoru ayırır."""
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()

    if not text:
        return np.nan, np.nan

    match = re.fullmatch(
        r"(\d+)\s*[-–:]\s*(\d+)",
        text
    )

    if not match:
        return np.nan, np.nan

    return int(match.group(1)), int(match.group(2))


def main():
    if not XLSX.exists():
        raise FileNotFoundError(
            f"Excel dosyası bulunamadı: {XLSX}"
        )

    raw = pd.read_excel(
        XLSX,
        header=None,
        sheet_name=0,
        engine="openpyxl"
    )

    required_columns = ODDS_START + len(FEATURES)

    if raw.shape[1] < required_columns:
        raise ValueError(
            f"Excel dosyasında yeterli sütun yok. "
            f"Beklenen en az {required_columns}, "
            f"bulunan {raw.shape[1]}."
        )

    # Temel maç bilgileri.
    data = pd.DataFrame({
        name: raw.iloc[8:, index].reset_index(drop=True)
        for name, index in COLS.items()
    })

    # İlk yarı ve maç sonu skorlarını ayrıştır.
    ht_scores = data["score_ht"].map(parse_score)
    ms_scores = data["score_ms"].map(parse_score)

    data["ht_home"] = ht_scores.map(lambda x: x[0])
    data["ht_away"] = ht_scores.map(lambda x: x[1])

    data["ft_home"] = ms_scores.map(lambda x: x[0])
    data["ft_away"] = ms_scores.map(lambda x: x[1])

    data.drop(columns=["score_ht", "score_ms"], inplace=True)

    # 39 oranı doğru Excel sütunlarından oku.
    for feature_index, excel_index in zip(
        range(36, 75),
        range(ODDS_START, ODDS_START + len(FEATURES))
    ):
        column = f"odds_{feature_index}"

        data[column] = (
            raw.iloc[8:, excel_index]
            .reset_index(drop=True)
            .map(parse_odds)
        )

    # Tarih ve sayısal alanlar.
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    ).dt.normalize()

    for column in [
        "ht_home", "ht_away",
        "ft_home", "ft_away"
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Metin alanları.
    for column in ["home", "away", "league", "time"]:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    data = data.dropna(subset=["date"]).copy()

    # Maç sonucu hedefi.
    data["target_ms"] = np.select(
        [
            data.ft_home > data.ft_away,
            data.ft_home == data.ft_away,
            data.ft_home < data.ft_away,
        ],
        [1, 0, 2],
        default=np.nan,
    )

    # 2,5 üst.
    data["target_o25"] = np.where(
        data.ft_home.notna() & data.ft_away.notna(),
        ((data.ft_home + data.ft_away) > 2.5).astype(float),
        np.nan
    )

    # İlk yarı 1,5 üst.
    data["target_ht15"] = np.where(
        data.ht_home.notna() & data.ht_away.notna(),
        ((data.ht_home + data.ht_away) > 1.5).astype(float),
        np.nan
    )

    # Karşılıklı gol.
    data["target_btts"] = np.where(
        data.ft_home.notna() & data.ft_away.notna(),
        (
            (data.ft_home > 0) &
            (data.ft_away > 0)
        ).astype(float),
        np.nan
    )

    JSON.parent.mkdir(parents=True, exist_ok=True)

    data.to_json(
        JSON,
        orient="records",
        date_format="iso",
        force_ascii=False
    )

    # Aktarım kontrolü.
    ht_count = (
        data.ht_home.notna() & data.ht_away.notna()
    ).sum()

    ft_count = (
        data.ft_home.notna() & data.ft_away.notna()
    ).sum()

    odds_count = data[FEATURES].notna().sum(axis=1).gt(0).sum()

    print("----------------------------------------")
    print(f"Aktarılan maç: {len(data):,}")
    print(f"İlk yarı skoru bulunan maç: {ht_count:,}")
    print(f"Maç sonu skoru bulunan maç: {ft_count:,}")
    print(f"Oranı bulunan maç: {odds_count:,}")
    print(f"JSON dosyası: {JSON}")
    print(f"Dosya boyutu: {JSON.stat().st_size:,} bytes")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
