
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

XLSX = ROOT / "input" / "Excel-Açılış-Cep-Telefonu.xlsx"
JSON = ROOT / "input" / "matches.json"

# Excel sütunları A'dan başlıyor.
# Python indeksleri sıfırdan başladığı için Excel sütun numarasından 1 çıkarılır.
COLS = {
    "score_ht": 21,  # V - SKOR İY
    "score_ms": 22,  # W - SKOR MS
    "date": 23,      # X - TARİH
    "league": 25,    # Z - LİG
    "time": 26,      # AA - SAAT
    "home": 27,      # AB - EV SAHİBİ
    "away": 28,      # AC - DEPLASMAN
}

# AD sütunundan başlayarak 39 oran sütunu.
ODDS_START = 29
FEATURES = [f"odds_{i}" for i in range(36, 75)]


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


def parse_score(value):
    """
    SKOR İY ve SKOR MS hücrelerini ayrıştırır.
    Desteklenen örnekler: 1-0, 1:0, 1 - 0, 1–0.
    """

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

    required_columns = ODDS_START + len(FEATURES)

    if raw.shape[1] < required_columns:
        raise ValueError(
            f"Excel sütun sayısı yetersiz. "
            f"Beklenen: en az {required_columns}, "
            f"bulunan: {raw.shape[1]}"
        )

    # Veriler 9. satırdan itibaren başlıyor.
    source = raw.iloc[8:].reset_index(drop=True)

    data = pd.DataFrame()

    # Maç bilgileri.
    for name, index in COLS.items():
        data[name] = source.iloc[:, index]

    # İlk yarı skorlarını ayrıştır.
    ht_scores = data["score_ht"].map(parse_score)

    data["ht_home"] = ht_scores.map(lambda score: score[0])
    data["ht_away"] = ht_scores.map(lambda score: score[1])

    # Maç sonu skorlarını ayrıştır.
    ft_scores = data["score_ms"].map(parse_score)

    data["ft_home"] = ft_scores.map(lambda score: score[0])
    data["ft_away"] = ft_scores.map(lambda score: score[1])

    data.drop(
        columns=["score_ht", "score_ms"],
        inplace=True
    )

    # Oranları AD sütunundan başlayarak oku.
    for offset, feature_index in enumerate(range(36, 75)):
        excel_index = ODDS_START + offset
        column = f"odds_{feature_index}"

        data[column] = (
            source.iloc[:, excel_index]
            .map(parse_odds)
        )

    # Tarih dönüşümü.
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    ).dt.normalize()

    # Skor alanlarını sayısal yap.
    score_columns = [
        "ht_home",
        "ht_away",
        "ft_home",
        "ft_away"
    ]

    for column in score_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Takım, lig ve saat bilgileri.
    for column in ["home", "away", "league", "time"]:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Tarihi olmayan satırları kaldır.
    data = data.dropna(subset=["date"]).copy()

    # Boş dosya oluşturulmasını engelle.
    if data.empty:
        raise ValueError(
            "Hiç maç aktarılamadı. "
            "Tarih sütunu X olarak ayarlandı ancak geçerli tarih bulunamadı. "
            "Excel'deki veri başlangıç satırını ve X sütununu kontrol et."
        )

    # Maç sonucu hedefi:
    # 1 = Ev sahibi, 0 = Beraberlik, 2 = Deplasman.
    data["target_ms"] = np.select(
        [
            data["ft_home"] > data["ft_away"],
            data["ft_home"] == data["ft_away"],
            data["ft_home"] < data["ft_away"],
        ],
        [1, 0, 2],
        default=np.nan
    )

    # 2,5 gol üstü.
    data["target_o25"] = np.where(
        data["ft_home"].notna() & data["ft_away"].notna(),
        (
            data["ft_home"] + data["ft_away"] > 2.5
        ).astype(float),
        np.nan
    )

    # İlk yarı 1,5 gol üstü.
    data["target_ht15"] = np.where(
        data["ht_home"].notna() & data["ht_away"].notna(),
        (
            data["ht_home"] + data["ht_away"] > 1.5
        ).astype(float),
        np.nan
    )

    # Karşılıklı gol.
    data["target_btts"] = np.where(
        data["ft_home"].notna() & data["ft_away"].notna(),
        (
            (data["ft_home"] > 0) &
            (data["ft_away"] > 0)
        ).astype(float),
        np.nan
    )

    # JSON kaydet.
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

    print("")
    print("========== AKTARIM RAPORU ==========")
    print(f"Toplam maç              : {len(data):,}")
    print(f"İlk yarı skoru bulunan  : {ht_count:,}")
    print(f"Maç sonu skoru bulunan  : {ft_count:,}")
    print(f"Oranı bulunan maç       : {odds_count:,}")
    print(f"JSON dosyası            : {JSON}")
    print(f"Dosya boyutu            : {JSON.stat().st_size:,} byte")
    print("====================================")


if __name__ == "__main__":
    main()
