
from pathlib import Path
import re
from datetime import datetime, date

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "input" / "acilis.xlsx"
JSON = ROOT / "input" / "matches.json"

FEATURES = [f"odds_{i}" for i in range(36, 75)]
ODDS_COUNT = len(FEATURES)


def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def parse_date(value):
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).normalize()

    if isinstance(value, (int, float, np.number)):
        number = float(value)

        if 20000 <= number <= 80000:
            return (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(number, unit="D")
            ).normalize()

        return pd.NaT

    text = str(value).strip()

    formats = [
        "%d.%m.%Y",
        "%d.%m.%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d.%m.%Y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return pd.Timestamp(
                datetime.strptime(text, fmt)
            ).normalize()
        except ValueError:
            pass

    result = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=True
    )

    return pd.Timestamp(result).normalize() if not pd.isna(result) else pd.NaT


def parse_score(value):
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()

    match = re.fullmatch(
        r"(\d+)\s*[-–—:]\s*(\d+)",
        text
    )

    if not match:
        return np.nan, np.nan

    return int(match.group(1)), int(match.group(2))


def parse_odds(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace(" ", "").replace(",", ".")

    try:
        return float(text)
    except (ValueError, TypeError):
        return np.nan


def find_header_row(raw):
    required = {
        "SKOR İY",
        "SKOR MS",
        "TARİH",
        "LİG",
        "SAAT",
        "EV SAHİBİ",
        "DEPLASMAN",
    }

    for row_index in range(min(40, len(raw))):
        values = {
            normalize(value)
            for value in raw.iloc[row_index].tolist()
        }

        if required.issubset(values):
            return row_index

    raise ValueError(
        "Excel başlık satırı bulunamadı. "
        "SKOR İY, SKOR MS, TARİH, LİG, SAAT, "
        "EV SAHİBİ ve DEPLASMAN başlıklarını kontrol et."
    )


def find_column(headers, name):
    target = normalize(name)

    for index, value in enumerate(headers):
        if normalize(value) == target:
            return index

    raise ValueError(f"Excel başlığı bulunamadı: {name}")


def main():
    if not XLSX.exists():
        raise FileNotFoundError(
            f"Excel dosyası bulunamadı: {XLSX}"
        )

    print("Excel dosyası okunuyor...")

    raw = pd.read_excel(
        XLSX,
        header=None,
        sheet_name=0,
        engine="openpyxl"
    )

    header_row = find_header_row(raw)
    headers = raw.iloc[header_row].tolist()

    print(f"Başlık satırı bulundu: Excel satırı {header_row + 1}")

    columns = {
        "score_ht": find_column(headers, "SKOR İY"),
        "score_ms": find_column(headers, "SKOR MS"),
        "date": find_column(headers, "TARİH"),
        "league": find_column(headers, "LİG"),
        "time": find_column(headers, "SAAT"),
        "home": find_column(headers, "EV SAHİBİ"),
        "away": find_column(headers, "DEPLASMAN"),
    }

    # Takım isimlerinden sonraki ilk "1" oran başlığını bul.
    away_index = columns["away"]
    odds_start = None

    for index in range(away_index + 1, len(headers)):
        if normalize(headers[index]) == "1":
            if index + ODDS_COUNT <= len(headers):
                odds_start = index
                break

    if odds_start is None:
        raise ValueError(
            "Takım isimlerinden sonra 39 oran sütunu bulunamadı. "
            "İlk oran başlığının '1' olduğunu kontrol et."
        )

    # Başlık satırının altındaki maç verileri.
    source = raw.iloc[header_row + 1:].reset_index(drop=True)

    data = pd.DataFrame()

    # Maç bilgileri.
    for name, index in columns.items():
        data[name] = source.iloc[:, index]

    # Skorları ayrıştır.
    ht_scores = data["score_ht"].map(parse_score)
    ms_scores = data["score_ms"].map(parse_score)

    data["ht_home"] = ht_scores.map(lambda x: x[0])
    data["ht_away"] = ht_scores.map(lambda x: x[1])

    data["ft_home"] = ms_scores.map(lambda x: x[0])
    data["ft_away"] = ms_scores.map(lambda x: x[1])

    data.drop(
        columns=["score_ht", "score_ms"],
        inplace=True
    )

    # Tarih.
    data["date"] = data["date"].map(parse_date)

    # Oranlar.
    for offset, feature_index in enumerate(range(36, 75)):
        excel_index = odds_start + offset

        data[f"odds_{feature_index}"] = (
            source.iloc[:, excel_index].map(parse_odds)
        )

    # Skorları sayısal hale getir.
    for column in [
        "ht_home",
        "ht_away",
        "ft_home",
        "ft_away",
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

    # Tarihi olmayan satırları kaldır.
    data = data.dropna(subset=["date"]).copy()

    if data.empty:
        raise ValueError(
            "Başlıklar bulundu ancak geçerli tarih içeren maç yok."
        )

    # MS sonucu.
    data["target_ms"] = np.select(
        [
            data["ft_home"] > data["ft_away"],
            data["ft_home"] == data["ft_away"],
            data["ft_home"] < data["ft_away"],
        ],
        [1, 0, 2],
        default=np.nan
    )

    # 2,5 Üst.
    data["target_o25"] = np.where(
        data["ft_home"].notna() & data["ft_away"].notna(),
        (
            data["ft_home"] + data["ft_away"] > 2.5
        ).astype(float),
        np.nan
    )

    # İY 1,5 Üst.
    data["target_ht15"] = np.where(
        data["ht_home"].notna() & data["ht_away"].notna(),
        (
            data["ht_home"] + data["ht_away"] > 1.5
        ).astype(float),
        np.nan
    )

    # KG Var.
    data["target_btts"] = np.where(
        data["ft_home"].notna() & data["ft_away"].notna(),
        (
            (data["ft_home"] > 0) &
            (data["ft_away"] > 0)
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

    ht_count = (
        data["ht_home"].notna() &
        data["ht_away"].notna()
    ).sum()

    ft_count = (
        data["ft_home"].notna() &
        data["ft_away"].notna()
    ).sum()

    odds_count = (
        data[FEATURES].notna().sum(axis=1) > 0
    ).sum()

    print("\n========== AKTARIM RAPORU ==========")
    print(f"Başlık satırı           : {header_row + 1}")
    print(f"Tarih sütunu            : {columns['date'] + 1}")
    print(f"İlk oran sütunu         : {odds_start + 1}")
    print(f"Toplam maç              : {len(data):,}")
    print(f"İlk yarı skoru bulunan  : {ht_count:,}")
    print(f"Maç sonu skoru bulunan  : {ft_count:,}")
    print(f"Oranı bulunan maç       : {odds_count:,}")
    print(f"JSON boyutu             : {JSON.stat().st_size:,} byte")
    print("Aktarım başarılı.")
    print("====================================")


if __name__ == "__main__":
    main()
