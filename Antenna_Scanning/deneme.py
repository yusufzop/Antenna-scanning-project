import os
import re
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIG
# ============================================================

BASE_SCENARIO_DIR = r"C:\Users\Bogazici\Desktop\senaryolar"

SCENARIO_EMITTER_RANGES = {
    1: range(0, 3),
    3: range(0, 6),
    4: range(0, 2),
    5: range(0, 3),
    6: range(0, 5),
    7: range(0, 5),
    8: range(0, 3),
    9: range(0, 6),
}


# ============================================================
# HELPERS
# ============================================================

def scenario_dir(scenario_id):
    return os.path.join(BASE_SCENARIO_DIR, f"senaryo {scenario_id}")


def scanning_file_for_scenario(scenario_id, emitter_id):
    direct = os.path.join(scenario_dir(scenario_id), f"emitter_{emitter_id}.txt")
    if os.path.exists(direct):
        return direct

    fallback = os.path.join(
        scenario_dir(scenario_id),
        "antenna_scanning",
        f"emitter_{emitter_id}.txt"
    )
    return fallback


def safe_int(x):
    try:
        if pd.isna(x):
            return np.nan
        return int(float(x))
    except Exception:
        return np.nan


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def split_table_line_2plus_spaces(line):
    return re.split(r"\s{2,}", line.strip())


def normalize_label(x):
    if pd.isna(x):
        return ""

    s = str(x).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def is_no_scanning_label(x):
    s = normalize_label(x)

    return s in {
        "no_scanning",
        "noscanning",
        "no_scan",
        "noscan",
        "none",
        "nan",
    }


def is_lock_on_label(x):
    s = normalize_label(x)

    return s in {
        "lock_on",
        "lockon",
        "lock",
    }


# ============================================================
# READ SCANNING FILES
# ============================================================

def read_single_scanning_file(file_path, scenario_id, emitter_id):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]

    if len(lines) < 3:
        return pd.DataFrame()

    header = split_table_line_2plus_spaces(lines[0])
    data_lines = lines[2:]

    rows = []

    for ln in data_lines:
        parts = split_table_line_2plus_spaces(ln)

        if len(parts) != len(header):
            continue

        row = dict(zip(header, parts))
        row["scenario_id"] = scenario_id
        row["emitter_id"] = emitter_id
        row["source_file"] = file_path
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    col_map = {
        "Tarama Tipi": "tarama_tipi",
        "Tarama idx": "tarama_idx",
        "Darbe Dizi idx": "pulse_train_id",
        "Amplitude": "amplitude",
    }

    missing = [c for c in col_map if c not in df.columns]
    if missing:
        print("[UYARI] Eksik kolon:", file_path, missing)
        return pd.DataFrame()

    keep_cols = list(col_map.keys()) + ["scenario_id", "emitter_id", "source_file"]
    df = df[keep_cols].rename(columns=col_map)

    df["scenario_id"] = df["scenario_id"].apply(safe_int)
    df["emitter_id"] = df["emitter_id"].apply(safe_int)
    df["tarama_idx"] = df["tarama_idx"].apply(safe_int)
    df["pulse_train_id"] = df["pulse_train_id"].apply(safe_int)
    df["amplitude"] = df["amplitude"].apply(safe_float)

    before = len(df)
    df = df[~df["tarama_tipi"].apply(is_no_scanning_label)].copy()
    removed = before - len(df)

    if removed > 0:
        print(f"[BILGI] {os.path.basename(file_path)}: {removed} adet No_Scanning satırı atıldı.")

    return df


def read_all_scanning_files():
    dfs = []

    for scenario_id, emitter_ids in SCENARIO_EMITTER_RANGES.items():
        print(f"\n--- Senaryo {scenario_id} okunuyor ---")

        for emitter_id in emitter_ids:
            fp = scanning_file_for_scenario(scenario_id, emitter_id)

            if not os.path.exists(fp):
                print("[UYARI] Yok:", fp)
                continue

            df_e = read_single_scanning_file(fp, scenario_id, emitter_id)

            if not df_e.empty:
                dfs.append(df_e)
                print(f"Emitter {emitter_id}: {len(df_e)} scanning satırı okundu.")

    if not dfs:
        raise RuntimeError("Hiçbir emitter scanning dosyası okunamadı.")

    df = pd.concat(dfs, ignore_index=True)

    df = df.drop_duplicates(
        subset=["scenario_id", "emitter_id", "pulse_train_id"],
        keep="first"
    ).reset_index(drop=True)

    return df


# ============================================================
# LOCK-ON COUNT
# ============================================================

def main():
    scan_df = read_all_scanning_files()

    print("\nToplam scanning satırı:", len(scan_df))

    lock_on_df = scan_df[scan_df["tarama_tipi"].apply(is_lock_on_label)].copy()

    print("Lock-on scanning satırı:", len(lock_on_df))

    if lock_on_df.empty:
        print("\n[UYARI] Hiç lock-on scanning bulunamadı.")
        print("Mevcut tarama tipleri:")
        print(scan_df["tarama_tipi"].value_counts(dropna=False))
        return

    counts = (
        lock_on_df
        .groupby(["scenario_id", "emitter_id", "tarama_idx", "tarama_tipi"], dropna=False)
        .agg(
            pulse_train_count=("pulse_train_id", "nunique"),
            first_pulse_train_id=("pulse_train_id", "min"),
            last_pulse_train_id=("pulse_train_id", "max"),
            mean_amplitude=("amplitude", "mean"),
            min_amplitude=("amplitude", "min"),
            max_amplitude=("amplitude", "max"),
        )
        .reset_index()
        .sort_values(["scenario_id", "emitter_id", "tarama_idx"])
        .reset_index(drop=True)
    )

    print("\n========== LOCK-ON SCANNING PULSE TRAIN COUNTS ==========")
    print(counts.to_string(index=False))

    print("\n========== ÖZET ==========")
    print("Lock-on scan cluster sayısı:", len(counts))
    print("Toplam lock-on pulse train sayısı:", int(counts["pulse_train_count"].sum()))
    print("Ortalama pulse train / lock-on scan:", counts["pulse_train_count"].mean())
    print("Min pulse train / lock-on scan:", counts["pulse_train_count"].min())
    print("Max pulse train / lock-on scan:", counts["pulse_train_count"].max())

    print("\n========== PULSE TRAIN COUNT DAĞILIMI ==========")
    print(counts["pulse_train_count"].describe())

    print("\n========== EN SIK GÖRÜLEN LOCK-ON UZUNLUKLARI ==========")
    print(counts["pulse_train_count"].value_counts().sort_index())


if __name__ == "__main__":
    main()