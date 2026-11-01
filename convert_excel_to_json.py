
from pathlib import Path
import re
from datetime import datetime, date

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "input" / "Excel-Açılış-Cep-Telefonu.xlsx"
JSON = ROOT / "input" / "matches.json"

# Excel sütunları A'dan başlıyor.
COLS = {
    "score_ht": 21,  # V - SKOR İY
    "score_ms": 22,  # W - SKOR MS
    "date": 23,      # X - TARİH
    "league": 25,    # Z - LİG
    "time": 26,      # AA - SAAT
    "home": 27,      # AB - EV SAHİBİ
    "away": 28,      # AC - DEPLASMAN
}

# Mevcut model özellikleri aynen korunuyor.
FEATURES = [f"odds_{i}" for i in range(36, 75)]


def parse_date(value):
    """Excel tarihlerini ve farklı metin tarih biçimlerini okur."""

    if pd.isna(value):
        return pd.NaT

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).normalize()

    if isinstance(value, (int, float, np.number)):
        number = float(value)

        # Excel seri tarih numarası.
        if 20000 <= number <= 80000:
            return (
                pd.Timestamp("1899-12-30") +
                pd.to_timedelta(number, unit="D")
            ).normalize()

        return pd.NaT

    text = str(value).strip()

    if not text:
        return pd.NaT

    formats = [
        "%d.%m.%Y",
        "%d.%m.%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d.%m.%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return pd.Timestamp(
                datetime.strptime(text, fmt)
            ).normalize()
        except ValueError:
            continue

    try:
        result = pd.to_datetime(
            text,
            dayfirst=True,
            errors="coerce"
        )

        if pd.isna(result):
            return pd.NaT

        return pd.Timestamp(result).normalize()

    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def parse_score(value):
    """1-0, 1:0, 1 - 0 biçimindeki skorları ayırır."""

    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()

    if not text:
        return np.nan, np.nan

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

    text = str(value).strip()
    text = text.replace(" ", "").replace(",", ".")

    try:
        return float(text)
    except (ValueError, TypeError):
        return np.nan


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

    if raw.shape[1] < 75:
        raise ValueError(
            f"Excel sütun sayısı yetersiz. "
            f"Beklenen: en az 75, bulunan: {raw.shape[1]}"
        )

    # Mevcut dosya yapısındaki gibi 9. satırdan itibaren veriler.
    source = raw.iloc[8:].reset_index(drop=True)

    data = pd.DataFrame()

    # Maç bilgileri.
    for name, index in COLS.items():
        data[name] = source.iloc[:, index]

    # İlk yarı skorları.
    ht_scores = data["score_ht"].map(parse_score)

    data["ht_home"] = ht_scores.map(lambda x: x[0])
    data["ht_away"] = ht_scores.map(lambda x: x[1])

    # Maç sonu skorları.
    ft_scores = data["score_ms"].map(parse_score)

    data["ft_home"] = ft_scores.map(lambda x: x[0])
    data["ft_away"] = ft_scores.map(lambda x: x[1])

    data.drop(
        columns=["score_ht", "score_ms"],
        inplace=True
    )

    # Modelin mevcut odds_36 - odds_74 sütunları.
    for index in range(36, 75):
        data[f"odds_{index}"] = (
            source.iloc[:, index].map(parse_odds)
        )

    # Tarihleri güvenli biçimde dönüştür.
    data["date"] = data["date"].map(parse_date)

    valid_dates = data["date"].notna().sum()

    if valid_dates == 0:
        samples = raw.iloc[8:, 23].head(15).tolist()

        raise ValueError(
            "Tarih sütunundan geçerli tarih okunamadı. "
            f"X sütunundaki örnek değerler: {samples}"
        )

    # Skorları sayısal hale getir.
    for column in [
        "ht_home",
        "ht_away",
        "ft_home",
        "ft_away"
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
        raise ValueError("Geçerli maç kaydı bulunamadı.")

    # MS sonucu: 1 ev sahibi, 0 beraberlik, 2 deplasman.
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

    # JSON oluştur.
    JSON.parent.mkdir(parents=True, exist_ok=True)

    data.to_json(
        JSON,
        orient="records",
        date_format="iso",
        force_ascii=False
    )

    # Aktarım raporu.
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
    print(f"Toplam maç              : {len(data):,}")
    print(f"Geçerli tarih           : {valid_dates:,}")
    print(f"İlk yarı skoru bulunan  : {ht_count:,}")
    print(f"Maç sonu skoru bulunan  : {ft_count:,}")
    print(f"Oranı bulunan maç       : {odds_count:,}")
    print(f"JSON boyutu             : {JSON.stat().st_size:,} byte")
    print("====================================")
    print("Aktarım başarılı.")


if __name__ == "__main__":
    main()
