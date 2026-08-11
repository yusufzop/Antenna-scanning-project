import os
import re
import glob
import math
import random
import pickle
import json
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ============================================================
# 1) PATH CONFIG  - mevcut koddaki path yapısı kullanıldı
# ============================================================

BASE_SCENARIO_DIR = r"C:\Users\Bogazici\Desktop\senaryolar"
BASE_SYNTHETIC_DIR = r"C:\Users\Bogazici\Desktop\kodlar"

SYNTHETIC_TRAIN_DIRS = {
    "circular_scanning": os.path.join(BASE_SYNTHETIC_DIR, "generated_circular_1200_txt"),
    "raster_scanning": os.path.join(BASE_SYNTHETIC_DIR, "generated_raster_1200_txt"),
    "conical_scanning": os.path.join(BASE_SYNTHETIC_DIR, "generated_conical_1200_txt"),
    "lock_on": os.path.join(BASE_SYNTHETIC_DIR, "generated_lock_on_1200_txt"),
}

SCENARIO_CLUSTER_OUTPUT_DIRS = {
    scenario_id: os.path.join(BASE_SCENARIO_DIR, f"scan_tour_clustering_outputs{scenario_id}")
    for scenario_id in range(1, 10)
}

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

FEATURE_FOLDER_NAME = "Darbe Dizi özellikleri"

# Output yeri özellikle değiştirildi: ResNet çıktısının üstüne yazmaz.
OUTPUT_DIR = os.path.join(BASE_SCENARIO_DIR, "scan_type_lpvg_outputs_synthetic_train_scenario_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2) CONFIG
# ============================================================

SEED = 42
MIN_TRAINS_PER_SCAN = 3
REAL_VAL_RATIO = 0.40

# Makaledeki LPVG gösteriminde M=1 örneği veriliyor.
# İstersen tek değer yerine (0, 1, 2) yapıp VG+LPVG featurelarını beraber kullanabilirsin.
DEFAULT_LPVG_M_VALUES = (1,)
LPVG_M_CANDIDATES = (0, 1, 2)
LPVG_M_VALUE_SETS = (
    (0,),
    (1,),
    (2,),
    (0, 1),
    (1, 2),
    (0, 1, 2),
)

# Çok uzun dizilerde LPVG O(N^3)'e yaklaşabilir. Bu limit hız için var.
# 0/None yaparsan kırpmaz.
MAX_POINTS_PER_SEQUENCE = 450

# SVM ayarları. RBF kernel küçük/orta feature setlerinde genelde daha mantıklı.
DEFAULT_SVM_PARAMS = {
    "C": 10.0,
    "gamma": "scale",
    "kernel": "rbf",
}
SVM_PARAM_GRID = (
    {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
    {"C": 3.0, "gamma": "scale", "kernel": "rbf"},
    {"C": 10.0, "gamma": "scale", "kernel": "rbf"},
    {"C": 30.0, "gamma": "scale", "kernel": "rbf"},
    {"C": 10.0, "gamma": "auto", "kernel": "rbf"},
    {"C": 1.0, "gamma": "scale", "kernel": "linear"},
    {"C": 10.0, "gamma": "scale", "kernel": "linear"},
)


# ============================================================
# 3) GENEL HELPERS + ESKİ KODDAN OKUMA FONKSİYONLARI
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_int(x):
    try:
        if pd.isna(x):
            return np.nan
        return int(float(x))
    except Exception:
        return np.nan


def split_whitespace_line(line):
    return line.strip().split()


def split_table_line_2plus_spaces(line):
    return re.split(r"\s{2,}", line.strip())


def parse_frame_time_and_freq(filename):
    base = os.path.basename(filename)

    m = re.search(r"db_(\d+)_([0-9.]+)\s*MHz", base, flags=re.IGNORECASE)
    if m:
        frame_time = int(m.group(1))
        file_freq_mhz = float(m.group(2))
        return frame_time, file_freq_mhz

    m2 = re.search(r"_(\d+)_", base)
    frame_time = int(m2.group(1)) if m2 else np.nan

    m3 = re.search(r"_([0-9.]+)\s*MHz", base, flags=re.IGNORECASE)
    file_freq_mhz = float(m3.group(1)) if m3 else np.nan

    return frame_time, file_freq_mhz


def normalize_mode(mode_str):
    if pd.isna(mode_str):
        return np.nan

    m = re.search(r"mode[_\s-]*(\d+)", str(mode_str), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return safe_int(mode_str)


def is_no_scanning_label(x):
    if pd.isna(x):
        return False

    s = str(x).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)

    return s in {
        "no_scanning", "noscanning", "no_scan", "noscan", "none", "nan"
    }


def scenario_dir(scenario_id):
    return os.path.join(BASE_SCENARIO_DIR, f"senaryo {scenario_id}")


def feature_dir_for_scenario(scenario_id):
    return os.path.join(scenario_dir(scenario_id), FEATURE_FOLDER_NAME)


def scanning_file_for_scenario(scenario_id, emitter_id):
    direct = os.path.join(scenario_dir(scenario_id), f"emitter_{emitter_id}.txt")
    if os.path.exists(direct):
        return direct

    fallback = os.path.join(scenario_dir(scenario_id), "antenna_scanning", f"emitter_{emitter_id}.txt")
    return fallback


def read_single_feature_file(file_path, scenario_id):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        return None

    header_line = lines[0].replace("#", "").strip()
    value_line = lines[1].strip()

    header = split_whitespace_line(header_line)
    values = split_whitespace_line(value_line)

    if len(header) != len(values):
        print("[UYARI] Kolon sayısı uyuşmadı, geçiliyor:", os.path.basename(file_path))
        return None

    data = dict(zip(header, values))
    filename = os.path.basename(file_path)
    frame_time, file_freq_mhz = parse_frame_time_and_freq(filename)

    return {
        "scenario_id": scenario_id,
        "filename": filename,
        "frame_time": frame_time,
        "file_freq_mhz": file_freq_mhz,
        "emitter_id": safe_int(data.get("emitter_id(GT)")),
        "operation_mode": data.get("operation_mode(GT)"),
        "operation_mode_id": normalize_mode(data.get("operation_mode(GT)")),
        "pulse_train_id": safe_int(data.get("Darbe_Dizi_no")),
        "num_pulses": safe_int(data.get("num_pulses")),
        "modulation": safe_int(data.get("modulation")),
        "pri_mod_type": safe_int(data.get("pri_mod_type")),
        "antenna_scan_type_raw": safe_int(data.get("antenna_scan_type")),
        "toad": safe_float(data.get("TOAD")),
        "pw": safe_float(data.get("mean_pw")),
        "fc": safe_float(data.get("mean_fc")),
        "bw": safe_float(data.get("mean_bw")),
        "mean_toa_diff": safe_float(data.get("mean_toa_diff")),
        "mean_snr_db": safe_float(data.get("mean_snr_db")),
    }


def read_all_feature_files(feature_dir, scenario_id):
    if not os.path.exists(feature_dir):
        print("[UYARI] Feature klasörü yok:", feature_dir)
        return pd.DataFrame()

    files = sorted(glob.glob(os.path.join(feature_dir, "*.txt")))
    rows = []

    for fp in files:
        row = read_single_feature_file(fp, scenario_id)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[UYARI] Darbe Dizi özellikleri boş/okunamadı:", feature_dir)
        return df

    return df.sort_values(
        ["scenario_id", "frame_time", "emitter_id", "fc", "pulse_train_id"],
        na_position="last"
    ).reset_index(drop=True)


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
        print("[UYARI] Eksik kolon:", os.path.basename(file_path), missing)

    keep_cols = [c for c in col_map if c in df.columns] + ["scenario_id", "emitter_id"]
    df = df[keep_cols].rename(columns=col_map)

    df["scenario_id"] = df["scenario_id"].apply(safe_int)
    df["emitter_id"] = df["emitter_id"].apply(safe_int)
    df["pulse_train_id"] = df["pulse_train_id"].apply(safe_int)
    df["tarama_idx"] = df["tarama_idx"].apply(safe_int)
    df["amplitude"] = df["amplitude"].apply(safe_float)

    if "tarama_tipi" in df.columns:
        before = len(df)
        df = df[~df["tarama_tipi"].apply(is_no_scanning_label)].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"[BILGI] {os.path.basename(file_path)}: {removed} adet No_Scanning satırı atıldı.")

    return df


def read_all_scanning_files(scenario_id, emitter_ids):
    dfs = []

    for emitter_id in emitter_ids:
        fp = scanning_file_for_scenario(scenario_id, emitter_id)
        if not os.path.exists(fp):
            print("[UYARI] Yok:", fp)
            continue

        df_e = read_single_scanning_file(fp, scenario_id, emitter_id)
        if not df_e.empty:
            dfs.append(df_e)

    if not dfs:
        print(f"[UYARI] Senaryo {scenario_id} için scanning/emitter dosyalarından veri okunamadı.")
        return pd.DataFrame()

    scan_df = pd.concat(dfs, ignore_index=True)
    scan_df = scan_df.drop_duplicates(
        subset=["scenario_id", "emitter_id", "pulse_train_id"],
        keep="first"
    ).reset_index(drop=True)

    return scan_df


def read_all_scenarios():
    feature_dfs = []
    scan_dfs = []

    for scenario_id, emitter_ids in SCENARIO_EMITTER_RANGES.items():
        print(f"\n--- Senaryo {scenario_id} okunuyor ---")

        fdir = feature_dir_for_scenario(scenario_id)
        feature_df_s = read_all_feature_files(fdir, scenario_id)
        scan_df_s = read_all_scanning_files(scenario_id, emitter_ids)

        if not feature_df_s.empty:
            feature_dfs.append(feature_df_s)
            print(f"Feature train sayısı: {len(feature_df_s)}")

        if not scan_df_s.empty:
            scan_dfs.append(scan_df_s)
            print(f"Amplitude/scanning satır sayısı: {len(scan_df_s)}")

    if not feature_dfs:
        raise RuntimeError("Hiçbir senaryodan Darbe Dizi özellikleri okunamadı.")
    if not scan_dfs:
        raise RuntimeError("Hiçbir senaryodan emitter/amplitude bilgisi okunamadı.")

    return pd.concat(feature_dfs, ignore_index=True), pd.concat(scan_dfs, ignore_index=True)


def build_merged_dataset():
    feature_df, scan_df = read_all_scenarios()

    merged = feature_df.merge(
        scan_df,
        on=["scenario_id", "emitter_id", "pulse_train_id"],
        how="left",
        validate="many_to_one"
    )

    if "tarama_tipi" in merged.columns:
        before = len(merged)
        merged = merged[~merged["tarama_tipi"].apply(is_no_scanning_label)].copy()
        removed = before - len(merged)
        if removed > 0:
            print(f"[BILGI] Merged içinden {removed} adet No_Scanning train atıldı.")

    merged["gt_scan_cluster"] = np.where(
        merged["tarama_idx"].notna(),
        "scenario_" + merged["scenario_id"].astype("Int64").astype(str)
        + "_emitter_" + merged["emitter_id"].astype("Int64").astype(str)
        + "_scan_" + merged["tarama_idx"].astype("Int64").astype(str),
        np.nan
    )

    merged = merged.sort_values(
        ["scenario_id", "frame_time", "emitter_id", "fc", "pulse_train_id"],
        na_position="last"
    ).reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, "merged_for_lpvg_classification.txt")
    merged.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")

    print("\nMerged kaydedildi:", out_path)
    print("Toplam train:", len(merged))
    print("Senaryo sayısı:", merged["scenario_id"].nunique())
    print("Amplitude eksik:", merged["amplitude"].isna().sum())
    print("Tarama tipi eksik:", merged["tarama_tipi"].isna().sum())
    print("GT scan cluster eksik:", merged["gt_scan_cluster"].isna().sum())

    return merged


def read_generated_sample_file(file_path, label_str, cluster_id):
    rows = []
    in_table = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.lower() == "index gain":
                in_table = True
                continue

            if not in_table or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            frame_time = safe_int(parts[0])
            amplitude = safe_float(parts[1])

            if pd.isna(frame_time) or pd.isna(amplitude):
                continue

            rows.append({
                "source": "synthetic",
                "scenario_id": -1,
                "frame_time": frame_time,
                "pulse_train_id": frame_time,
                "emitter_id": -1,
                "tarama_tipi": label_str,
                "amplitude": amplitude,
                "gt_scan_cluster": cluster_id,
            })

    return rows


def build_synthetic_training_dataset():
    rows = []

    for label_str, folder in SYNTHETIC_TRAIN_DIRS.items():
        if not os.path.exists(folder):
            print("[UYARI] Synthetic klasor yok:", folder)
            continue

        files = sorted(glob.glob(os.path.join(folder, "*.txt")))

        for sample_idx, file_path in enumerate(files):
            cluster_id = f"synthetic_{label_str}_{sample_idx:05d}"
            rows.extend(read_generated_sample_file(file_path, label_str, cluster_id))

        print(f"Synthetic {label_str}: {len(files)} sample okundu.")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Synthetic training verisi okunamadi. generated_*_1200_txt pathlerini kontrol et.")

    out_path = os.path.join(OUTPUT_DIR, "synthetic_train_val_rows.txt")
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
    print("\nSynthetic train/val rows kaydedildi:", out_path)

    return df


def build_scenario_cluster_test_dataset():
    dfs = []

    for scenario_id, folder in SCENARIO_CLUSTER_OUTPUT_DIRS.items():
        file_path = os.path.join(folder, "clustered_pulse_train_features.txt")

        if not os.path.exists(file_path):
            print("[UYARI] Clustered scenario dosyasi yok:", file_path)
            continue

        df = pd.read_csv(file_path, sep="\t", encoding="utf-8-sig")

        if "pred_cluster" not in df.columns:
            print("[UYARI] pred_cluster kolonu yok:", file_path)
            continue

        # Cluster output dosyasında amplitude ve frame_time olmalı.
        # Kolon adları farklıysa aşağıdaki aliaslar yakalar.
        df = normalize_cluster_columns(df)

        df["source"] = "scenario_test"
        df["scenario_id"] = scenario_id
        df["gt_scan_cluster"] = (
            "scenario_" + str(scenario_id)
            + "_pred_cluster_" + df["pred_cluster"].astype("Int64").astype(str)
        )

        dfs.append(df)
        print(f"Scenario {scenario_id} clustered test rows: {len(df)}")

    if not dfs:
        raise RuntimeError("Scenario test verisi okunamadi. scan_tour_clustering_outputs1-9 pathlerini kontrol et.")

    test_df = pd.concat(dfs, ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, "scenario_cluster_test_rows.txt")
    test_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
    print("\nScenario cluster test rows kaydedildi:", out_path)

    return test_df


def normalize_cluster_columns(df):
    rename_map = {}

    if "amplitude" not in df.columns:
        for c in ["Amplitude", "amp", "gain", "Gain", "pa", "PA"]:
            if c in df.columns:
                rename_map[c] = "amplitude"
                break

    if "frame_time" not in df.columns:
        for c in ["Frame_Time", "time", "toa", "TOA", "index", "Index", "pulse_train_id", "Darbe_Dizi_no"]:
            if c in df.columns:
                rename_map[c] = "frame_time"
                break

    if "pulse_train_id" not in df.columns:
        for c in ["Darbe_Dizi_no", "Darbe Dizi idx", "pulse_train_idx", "train_id"]:
            if c in df.columns:
                rename_map[c] = "pulse_train_id"
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ============================================================
# 4) LPVG: TIME SERIES -> GRAPH MATRIX
# ============================================================

def normalize_01(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x, dtype=np.float64)

    mn = np.nanmin(x)
    mx = np.nanmax(x)

    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x, dtype=np.float64)

    return (x - mn) / (mx - mn)


def prepare_sequence(toa_values, amp_values, max_points=MAX_POINTS_PER_SEQUENCE):
    t = np.asarray(toa_values, dtype=np.float64)
    x = np.asarray(amp_values, dtype=np.float64)

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]

    if len(t) == 0:
        return t, x

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    # Aynı zaman değerleri varsa ortalıyoruz. LPVG'de t_i < t_c < t_b şartı için zamanlar tekil olmalı.
    tmp = pd.DataFrame({"t": t, "x": x}).groupby("t", as_index=False)["x"].mean()
    t = tmp["t"].to_numpy(dtype=np.float64)
    x = tmp["x"].to_numpy(dtype=np.float64)

    # Genlik ölçek değişimine çok takılmaması için 0-1 normalize ediyoruz.
    x = normalize_01(x)

    if max_points and len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points).round().astype(int)
        t = t[idx]
        x = x[idx]

    return t, x


def lpvg_adjacency(toa_values, amp_values, M=1):
    """
    Limited Penetrable Visibility Graph.

    İki nokta arasındaki doğru, aradaki histogram barlarını kaç kere kesiyorsa
    o sayı blockers olarak sayılır. blockers <= M ise edge kurulur.
    M=0 normal visibility graph'e karşılık gelir.
    """
    t, x = prepare_sequence(toa_values, amp_values)
    n = len(x)

    A = np.zeros((n, n), dtype=np.uint8)
    if n < 2:
        return A

    for i in range(n - 1):
        ti = t[i]
        xi = x[i]

        for j in range(i + 1, n):
            tj = t[j]
            xj = x[j]

            if j == i + 1:
                A[i, j] = 1
                A[j, i] = 1
                continue

            denom = tj - ti
            if abs(denom) < 1e-12:
                continue

            blockers = 0
            slope = (xj - xi) / denom

            for k in range(i + 1, j):
                line_y = xi + slope * (t[k] - ti)

                # x[k] >= line_y ise aradaki bar görüş çizgisini kesiyor.
                # Küçük epsilon eşitlik gürültüsünü azaltır.
                if x[k] >= line_y - 1e-12:
                    blockers += 1
                    if blockers > M:
                        break

            if blockers <= M:
                A[i, j] = 1
                A[j, i] = 1

    return A


# ============================================================
# 5) LPVG FEATURE EXTRACTION
# ============================================================

def average_degree(A):
    n = A.shape[0]
    if n == 0:
        return 0.0
    return float(A.sum(axis=1).mean())


def clustering_coefficient(A):
    n = A.shape[0]
    if n == 0:
        return 0.0

    deg = A.sum(axis=1).astype(int)
    coeffs = []

    for i in range(n):
        d = deg[i]
        if d < 2:
            coeffs.append(0.0)
            continue

        neigh = np.flatnonzero(A[i])
        sub = A[np.ix_(neigh, neigh)]
        actual_edges_between_neighbors = sub.sum() / 2.0
        possible_edges = d * (d - 1) / 2.0
        coeffs.append(actual_edges_between_neighbors / possible_edges)

    return float(np.mean(coeffs))


def normalized_information_entropy(A):
    n = A.shape[0]
    if n <= 2:
        return 0.0

    deg = A.sum(axis=1).astype(np.float64)
    deg = np.maximum(deg, 1.0)

    inv_d = 1.0 / deg
    E = -float(np.sum(inv_d * np.log(inv_d)))

    # Makaledeki teorik üst-alt sınırlar.
    Emax = -n * (1.0 / (n - 1.0)) * math.log(1.0 / (n - 1.0))
    Emin = -(n - 2.0) * 0.5 * math.log(0.5)

    denom = Emax - Emin
    if abs(denom) < 1e-12:
        return 0.0

    Enorm = (E - Emin) / denom
    return float(np.clip(Enorm, 0.0, 1.0))


def lpvg_complexity(A):
    n = A.shape[0]
    if n <= 2:
        return 0.0

    # A simetrik olduğu için eigvalsh daha stabil/hızlı.
    vals = np.linalg.eigvalsh(A.astype(np.float64))
    rmax = float(vals[-1])

    lower = 2.0 * math.cos(math.pi / (n + 1.0))
    upper = n - 1.0
    denom = upper - lower

    if abs(denom) < 1e-12:
        return 0.0

    rnorm = (rmax - lower) / denom
    rnorm = float(np.clip(rnorm, 0.0, 1.0))

    R = 4.0 * rnorm * (1.0 - rnorm)
    return float(np.clip(R, 0.0, 1.0))


def edge_density(A):
    # Makalede ana 4 feature var; density debug/yorum için ekleniyor ama SVM'e sokmuyoruz.
    n = A.shape[0]
    if n <= 1:
        return 0.0
    return float(A.sum() / (n * (n - 1)))


def extract_lpvg_features(toa_values, amp_values, m_values=DEFAULT_LPVG_M_VALUES):
    feature_dict = {}

    t, x = prepare_sequence(toa_values, amp_values)
    feature_dict["n_points"] = len(x)

    if len(x) == 0:
        for M in m_values:
            feature_dict[f"M{M}_avg_degree"] = 0.0
            feature_dict[f"M{M}_clustering_coeff"] = 0.0
            feature_dict[f"M{M}_info_entropy_norm"] = 0.0
            feature_dict[f"M{M}_complexity"] = 0.0
        return feature_dict

    for M in m_values:
        A = lpvg_adjacency(t, x, M=M)

        feature_dict[f"M{M}_avg_degree"] = average_degree(A)
        feature_dict[f"M{M}_clustering_coeff"] = clustering_coefficient(A)
        feature_dict[f"M{M}_info_entropy_norm"] = normalized_information_entropy(A)
        feature_dict[f"M{M}_complexity"] = lpvg_complexity(A)

    return feature_dict


def save_lpvg_matrix_image(toa_values, amp_values, out_png, M=1):
    # Debug için: makaledeki Fig. 9 gibi binary matrix resmi.
    try:
        from PIL import Image
    except Exception:
        print("[UYARI] PIL yok, LPVG matrix image kaydedilemedi.")
        return

    A = lpvg_adjacency(toa_values, amp_values, M=M)
    if A.size == 0:
        return

    img = (A * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_png)


# ============================================================
# 6) ROWS -> SAMPLE FEATURE TABLE
# ============================================================

def build_sample_feature_table(
    df,
    require_label=True,
    name="dataset",
    m_values=DEFAULT_LPVG_M_VALUES,
):
    needed = ["gt_scan_cluster", "amplitude", "frame_time"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: eksik kolonlar: {missing}")

    work = df.dropna(subset=["gt_scan_cluster", "amplitude", "frame_time"]).copy()

    if require_label:
        if "tarama_tipi" not in work.columns:
            raise RuntimeError(f"{name}: tarama_tipi kolonu yok.")
        work = work.dropna(subset=["tarama_tipi"]).copy()
        work = work[~work["tarama_tipi"].apply(is_no_scanning_label)].copy()

    rows = []

    for cluster_id, g in work.groupby("gt_scan_cluster"):
        g = g.sort_values("frame_time").copy()

        if len(g) < MIN_TRAINS_PER_SCAN:
            continue

        toa_values = g["frame_time"].to_numpy(dtype=np.float64)
        amp_values = g["amplitude"].to_numpy(dtype=np.float64)

        feats = extract_lpvg_features(toa_values, amp_values, m_values=m_values)

        row = {
            "cluster_id": cluster_id,
            "pulse_count": len(g),
            **feats,
        }

        if require_label:
            labels = g["tarama_tipi"].astype(str)
            label_counts = labels.value_counts().sort_index().to_dict()
            row["label"] = str(labels.mode().iloc[0])
            row["pulse_train_label_counts"] = json.dumps(label_counts, sort_keys=True)

        # Bilgi/debug kolonları
        for c in ["scenario_id", "emitter_id", "pred_cluster"]:
            if c in g.columns:
                vals = g[c].dropna().unique()
                if len(vals) > 0:
                    row[c] = vals[0]

        rows.append(row)

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        raise RuntimeError(f"{name}: LPVG sample feature tablosu boş. Cluster/pulse sayısı/pathleri kontrol et.")

    return feature_df


def feature_columns_from_table(feature_df, m_values=DEFAULT_LPVG_M_VALUES):
    cols = []
    for M in m_values:
        cols.extend([
            f"M{M}_avg_degree",
            f"M{M}_clustering_coeff",
            f"M{M}_info_entropy_norm",
            f"M{M}_complexity",
        ])
    return cols


# ============================================================
# 7) TRAIN / VAL / TEST
# ============================================================

def stratified_split_indices(labels, val_ratio=REAL_VAL_RATIO, seed=SEED):
    labels = np.asarray(labels)
    rng = random.Random(seed)

    val_idx = []
    test_idx = []

    for label in sorted(set(labels)):
        inds = np.where(labels == label)[0].tolist()
        rng.shuffle(inds)

        n_val = int(round(len(inds) * val_ratio))
        if len(inds) >= 2:
            n_val = max(1, min(n_val, len(inds) - 1))
        else:
            n_val = 0

        val_idx.extend(inds[:n_val])
        test_idx.extend(inds[n_val:])

    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return val_idx, test_idx


def train_svm(train_features_df, feature_cols, svm_params=None):
    if svm_params is None:
        svm_params = DEFAULT_SVM_PARAMS

    X_train = train_features_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_features_df["label"].astype(str).to_numpy()

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=svm_params["C"],
            gamma=svm_params["gamma"],
            kernel=svm_params["kernel"],
            class_weight="balanced",
            probability=True,
            random_state=SEED,
        )),
    ])

    clf.fit(X_train, y_train)
    return clf


def parse_pulse_train_label_counts(row):
    if "pulse_train_label_counts" in row and pd.notna(row["pulse_train_label_counts"]):
        try:
            counts = json.loads(row["pulse_train_label_counts"])
            return {str(k): int(v) for k, v in counts.items()}
        except Exception:
            pass

    label = str(row["label"])
    return {label: int(row.get("pulse_count", 1))}


def pulse_train_accuracy_summary(result_df):
    rows = []
    totals = {}
    correct = {}

    for _, row in result_df.iterrows():
        pred = str(row["predicted_scan_type"])
        for true_label, count in parse_pulse_train_label_counts(row).items():
            totals[true_label] = totals.get(true_label, 0) + count
            if pred == true_label:
                correct[true_label] = correct.get(true_label, 0) + count

    for label in sorted(totals):
        total_count = totals[label]
        correct_count = correct.get(label, 0)
        rows.append({
            "scan_type": label,
            "correct_pulse_trains": correct_count,
            "total_pulse_trains": total_count,
            "accuracy_percent": 100.0 * correct_count / total_count if total_count else 0.0,
        })

    total_pulse_trains = sum(totals.values())
    total_correct = sum(correct.values())
    rows.append({
        "scan_type": "TOTAL",
        "correct_pulse_trains": total_correct,
        "total_pulse_trains": total_pulse_trains,
        "accuracy_percent": 100.0 * total_correct / total_pulse_trains if total_pulse_trains else 0.0,
    })

    return pd.DataFrame(rows)


def evaluate_and_save(clf, features_df, feature_cols, split_name, out_prefix):
    X = features_df[feature_cols].to_numpy(dtype=np.float64)
    y_true = features_df["label"].astype(str).to_numpy() if "label" in features_df.columns else None
    y_pred = clf.predict(X)

    result_df = features_df.copy()
    result_df["predicted_scan_type"] = y_pred

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        result_df["pred_confidence"] = proba.max(axis=1)

    if y_true is not None:
        result_df["is_correct"] = y_true == y_pred
        acc = accuracy_score(y_true, y_pred)
        print(f"\n========== {split_name} RESULTS ==========")
        print(f"Accuracy: {acc:.4f}")

        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

        print("\nConfusion matrix:")
        print(cm_df)

        print("\nClassification report:")
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        print(report)

        cm_path = out_prefix + "_confusion_matrix.txt"
        cm_df.to_csv(cm_path, sep="\t", encoding="utf-8-sig")
        print("Confusion matrix kaydedildi:", cm_path)

        report_path = out_prefix + "_classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print("Classification report kaydedildi:", report_path)

        pulse_summary_df = pulse_train_accuracy_summary(result_df)
        print("\nPulse train accuracy by scan type:")
        print(pulse_summary_df.to_string(index=False))

        pulse_summary_path = out_prefix + "_pulse_train_accuracy_by_scan_type.txt"
        pulse_summary_df.to_csv(pulse_summary_path, sep="\t", index=False, encoding="utf-8-sig")
        print("Pulse train accuracy summary kaydedildi:", pulse_summary_path)

    pred_path = out_prefix + "_predictions.txt"
    result_df.to_csv(pred_path, sep="\t", index=False, encoding="utf-8-sig")
    print("Tahminler kaydedildi:", pred_path)

    return result_df


def evaluate_accuracy(clf, features_df, feature_cols):
    X = features_df[feature_cols].to_numpy(dtype=np.float64)
    y_true = features_df["label"].astype(str).to_numpy()
    y_pred = clf.predict(X)
    return accuracy_score(y_true, y_pred)


def select_params_with_validation(train_features_df, val_features_df):
    rows = []
    best = None

    print("\nValidation ile parametre seÃ§imi baÅŸlÄ±yor...")

    for m_values in LPVG_M_VALUE_SETS:
        feature_cols = feature_columns_from_table(train_features_df, m_values=m_values)

        missing_cols = [
            c for c in feature_cols
            if c not in train_features_df.columns or c not in val_features_df.columns
        ]
        if missing_cols:
            print(f"[UYARI] M={m_values} eksik feature kolonlarÄ± nedeniyle atlandÄ±: {missing_cols}")
            continue

        for svm_params in SVM_PARAM_GRID:
            clf = train_svm(train_features_df, feature_cols, svm_params=svm_params)
            val_acc = evaluate_accuracy(clf, val_features_df, feature_cols)

            row = {
                "lpvg_m_values": ",".join(str(m) for m in m_values),
                "svm_C": svm_params["C"],
                "svm_gamma": svm_params["gamma"],
                "svm_kernel": svm_params["kernel"],
                "val_accuracy": val_acc,
                "num_features": len(feature_cols),
            }
            rows.append(row)

            print(
                f"  M={m_values} | SVM={svm_params} | "
                f"val_acc={val_acc:.4f}"
            )

            candidate = (val_acc, -len(feature_cols), row, clf, feature_cols, svm_params, m_values)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    if best is None:
        raise RuntimeError("Validation parametre seÃ§imi iÃ§in uygun aday bulunamadÄ±.")

    selection_df = pd.DataFrame(rows).sort_values(
        ["val_accuracy", "num_features"],
        ascending=[False, True]
    ).reset_index(drop=True)
    selection_path = os.path.join(OUTPUT_DIR, "validation_parameter_search.txt")
    selection_df.to_csv(selection_path, sep="\t", index=False, encoding="utf-8-sig")
    print("Validation parametre arama sonuÃ§larÄ± kaydedildi:", selection_path)

    _, _, best_row, best_clf, best_feature_cols, best_svm_params, best_m_values = best
    print(
        "\nSeÃ§ilen parametreler: "
        f"M={best_m_values}, SVM={best_svm_params}, "
        f"val_acc={best_row['val_accuracy']:.4f}"
    )

    return {
        "clf": best_clf,
        "feature_cols": best_feature_cols,
        "svm_params": dict(best_svm_params),
        "lpvg_m_values": tuple(best_m_values),
        "validation_accuracy": best_row["val_accuracy"],
        "search_results_path": selection_path,
    }


def save_feature_table(feature_df, name):
    path = os.path.join(OUTPUT_DIR, f"{name}_lpvg_features.txt")
    feature_df.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"{name} LPVG feature tablosu kaydedildi:", path)
    return path


def save_example_matrices(train_features_df, source_df, m_values=DEFAULT_LPVG_M_VALUES):
    matrix_dir = os.path.join(OUTPUT_DIR, "example_lpvg_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    # Her labeldan bir örnek matrix kaydet.
    if "label" not in train_features_df.columns:
        return

    for label, row in train_features_df.groupby("label").head(1).groupby("label"):
        cluster_id = row.iloc[0]["cluster_id"]
        g = source_df[source_df["gt_scan_cluster"] == cluster_id].sort_values("frame_time")
        if g.empty:
            continue

        out_png = os.path.join(matrix_dir, f"example_{label}_M{m_values[0]}.png")
        save_lpvg_matrix_image(
            g["frame_time"].to_numpy(dtype=np.float64),
            g["amplitude"].to_numpy(dtype=np.float64),
            out_png,
            M=m_values[0],
        )
        print("Örnek LPVG matrix kaydedildi:", out_png)


# ============================================================
# 8) MAIN
# ============================================================

def main():
    set_seed(SEED)

    print("\n========== LPVG + SVM ANTENNA SCAN TYPE RECOGNITION ==========")
    print("Output dir:", OUTPUT_DIR)
    print("LPVG M candidates:", LPVG_M_CANDIDATES)
    print("LPVG M value sets:", LPVG_M_VALUE_SETS)
    print("MAX_POINTS_PER_SEQUENCE:", MAX_POINTS_PER_SEQUENCE)

    print("\nVeriler okunuyor...")
    train_rows_df = build_synthetic_training_dataset()
    scenario_rows_df = build_scenario_cluster_test_dataset()

    print("\nLPVG feature extraction başlıyor...")
    train_features_df = build_sample_feature_table(
        train_rows_df,
        require_label=True,
        name="synthetic_train",
        m_values=LPVG_M_CANDIDATES,
    )

    # Scenario cluster dosyanızda gerçek tarama tipi varsa onu değerlendirir.
    # Yoksa sadece prediction üretmek için label gerektirmeden feature çıkarırız.
    scenario_has_label = "tarama_tipi" in scenario_rows_df.columns and scenario_rows_df["tarama_tipi"].notna().any()
    scenario_features_df = build_sample_feature_table(
        scenario_rows_df,
        require_label=scenario_has_label,
        name="scenario_cluster_test",
        m_values=LPVG_M_CANDIDATES,
    )

    save_feature_table(train_features_df, "synthetic_train")
    save_feature_table(scenario_features_df, "scenario_cluster_test")

    feature_cols = feature_columns_from_table(train_features_df, m_values=LPVG_M_CANDIDATES)

    print("\nFeature kolonları:")
    for c in feature_cols:
        print("  ", c)

    print("\nSynthetic train sample sayısı:", len(train_features_df))
    print("Class counts:")
    for k, v in Counter(train_features_df["label"]).items():
        print(" ", k, ":", v)

    print("\nSVM eğitiliyor...")
    val_df = None
    test_df = None
    selected_m_values = DEFAULT_LPVG_M_VALUES
    selected_svm_params = dict(DEFAULT_SVM_PARAMS)
    validation_accuracy = None

    if scenario_has_label:
        val_idx, test_idx = stratified_split_indices(scenario_features_df["label"].astype(str).to_numpy())
        val_df = scenario_features_df.iloc[val_idx].reset_index(drop=True)
        test_df = scenario_features_df.iloc[test_idx].reset_index(drop=True)

        print(
            f"\nReal scenario split: val={len(val_df)} ({REAL_VAL_RATIO:.0%}), "
            f"test={len(test_df)} ({1.0 - REAL_VAL_RATIO:.0%})"
        )

    if scenario_has_label and val_df is not None and len(val_df) > 0:
        selection = select_params_with_validation(train_features_df, val_df)
        clf = selection["clf"]
        feature_cols = selection["feature_cols"]
        selected_m_values = selection["lpvg_m_values"]
        selected_svm_params = selection["svm_params"]
        validation_accuracy = selection["validation_accuracy"]
    else:
        print("\n[UYARI] Label'li validation yok; default LPVG/SVM parametreleri kullanilacak.")
        feature_cols = feature_columns_from_table(train_features_df, m_values=selected_m_values)
        clf = train_svm(train_features_df, feature_cols, svm_params=selected_svm_params)

    print("\nSecilen LPVG M values:", selected_m_values)
    print("Secilen SVM parametreleri:", selected_svm_params)
    print("Secilen feature kolonlari:")
    for c in feature_cols:
        print("  ", c)
    save_example_matrices(train_features_df, train_rows_df, m_values=selected_m_values)

    model_path = os.path.join(OUTPUT_DIR, "lpvg_svm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_cols": feature_cols,
            "lpvg_m_values": selected_m_values,
            "svm_params": selected_svm_params,
            "validation_accuracy": validation_accuracy,
            "max_points_per_sequence": MAX_POINTS_PER_SEQUENCE,
        }, f)
    print("Model kaydedildi:", model_path)

    # Synthetic üzerinde sanity-check.
    evaluate_and_save(
        clf,
        train_features_df,
        feature_cols,
        split_name="SYNTHETIC TRAIN SANITY CHECK",
        out_prefix=os.path.join(OUTPUT_DIR, "synthetic_train")
    )

    if scenario_has_label:
        if len(val_df) > 0:
            evaluate_and_save(
                clf,
                val_df,
                feature_cols,
                split_name="REAL SCENARIO VAL",
                out_prefix=os.path.join(OUTPUT_DIR, "scenario_val")
            )

        if len(test_df) > 0:
            evaluate_and_save(
                clf,
                test_df,
                feature_cols,
                split_name="REAL SCENARIO TEST",
                out_prefix=os.path.join(OUTPUT_DIR, "scenario_test")
            )
    else:
        print("\n[UYARI] Scenario cluster dosyasında tarama_tipi/label yok gibi görünüyor.")
        print("Bu yüzden gerçek accuracy hesaplanmadı; sadece cluster tahmini kaydedilecek.")
        evaluate_and_save(
            clf,
            scenario_features_df,
            feature_cols,
            split_name="REAL SCENARIO PREDICTION ONLY",
            out_prefix=os.path.join(OUTPUT_DIR, "scenario_cluster")
        )

    print("\nBitti.")


if __name__ == "__main__":
    main()
