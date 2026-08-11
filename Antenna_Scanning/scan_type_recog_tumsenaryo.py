import os
import re
import glob
import math
import random
from collections import Counter

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim


# ============================================================
# 1) YENİ ÇOKLU SENARYO PATH'LERİ
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

# Senin verdiğin emitter aralıkları: inclusive, yani 0-2 => 0,1,2
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

OUTPUT_DIR = os.path.join(BASE_SCENARIO_DIR, "scan_type_resnet_outputs_synthetic_train_scenario_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def scenario_dir(scenario_id):
    return os.path.join(BASE_SCENARIO_DIR, f"senaryo {scenario_id}")


def feature_dir_for_scenario(scenario_id):
    return os.path.join(scenario_dir(scenario_id), FEATURE_FOLDER_NAME)


def scanning_file_for_scenario(scenario_id, emitter_id):
    """
    Senin yeni yapıda emitter dosyaları şurada: 
        C:/Users/Bogazici/Desktop/senaryolar/senaryo X/emitter_Y.txt

    Eski yapıdaki antenna_scanning alt klasörü varsa onu da fallback olarak kontrol ediyor.
    """
    direct = os.path.join(scenario_dir(scenario_id), f"emitter_{emitter_id}.txt")
    if os.path.exists(direct):
        return direct

    fallback = os.path.join(scenario_dir(scenario_id), "antenna_scanning", f"emitter_{emitter_id}.txt")
    return fallback


# ============================================================
# 2) CONFIG
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
MIN_TRAINS_PER_SCAN = 3
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 3) OKUMA FONKSİYONLARI
# ============================================================

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


def parse_frame_time_and_freq(filename):
    """
    Örnek:
        -5.0db_77_3650MHz.txt
    frame_time = 77
    file_freq_mhz = 3650
    """
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


def split_whitespace_line(line):
    return line.strip().split()


def split_table_line_2plus_spaces(line):
    return re.split(r"\s{2,}", line.strip())


def normalize_mode(mode_str):
    if pd.isna(mode_str):
        return np.nan

    m = re.search(r"mode[_\s-]*(\d+)", str(mode_str), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return safe_int(mode_str)


def is_no_scanning_label(x):
    """
    Tarama Tipi kolonunda No_Scanning / no_scanning / no scanning gibi
    değerler varsa bunları gerçek scan type olarak kullanmayacağız.
    """
    if pd.isna(x):
        return False

    s = str(x).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)

    return s in {
        "no_scanning",
        "noscanning",
        "no_scan",
        "noscan",
        "none",
        "nan",
    }


def read_single_feature_file(file_path, scenario_id):
    """
    Darbe Dizi özellikleri dosyası:
        # emitter_id(GT) operation_mode(GT) Darbe_Dizi_no ...
          0              mode_2             5924 ...
    """
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

    row = {
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

        # TOAD burada zaman feature'ı gibi kullanılacak.
        "toad": safe_float(data.get("TOAD")),

        "pw": safe_float(data.get("mean_pw")),
        "fc": safe_float(data.get("mean_fc")),
        "bw": safe_float(data.get("mean_bw")),
        "mean_toa_diff": safe_float(data.get("mean_toa_diff")),
        "mean_snr_db": safe_float(data.get("mean_snr_db")),
    }

    return row


def read_all_feature_files(feature_dir, scenario_id):
    files = sorted(glob.glob(os.path.join(feature_dir, "*.txt")))
    rows = []

    if not os.path.exists(feature_dir):
        print("[UYARI] Feature klasörü yok:", feature_dir)
        return pd.DataFrame()

    for fp in files:
        row = read_single_feature_file(fp, scenario_id)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("[UYARI] Darbe Dizi özellikleri boş/okunamadı:", feature_dir)
        return df

    df = df.sort_values(
        ["scenario_id", "frame_time", "emitter_id", "fc", "pulse_train_id"],
        na_position="last"
    ).reset_index(drop=True)

    return df


def read_single_scanning_file(file_path, scenario_id, emitter_id):
    """
    antenna_scanning/emitter_X.txt dosyası:
        Tarama Tipi        Tarama idx  Adım (°)  Darbe Dizi idx  ... Amplitude ...
        -----------------  ----------  --------  --------------  ...
        raster_scanning    3           3.96      1               ... 2.0157 ...
    """
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

    # No_Scanning gerçek anten tarama türü değil; training'e sokma.
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

    feature_df = pd.concat(feature_dfs, ignore_index=True)
    scan_df = pd.concat(scan_dfs, ignore_index=True)

    return feature_df, scan_df


def build_merged_dataset():
    feature_df, scan_df = read_all_scenarios()

    merged = feature_df.merge(
        scan_df,
        on=["scenario_id", "emitter_id", "pulse_train_id"],
        how="left",
        validate="many_to_one"
    )

    # Ek güvenlik: No_Scanning merge'e bir şekilde geldiyse tamamen çıkar.
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

    out_path = os.path.join(OUTPUT_DIR, "merged_for_scan_type_classification.txt")
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


# ============================================================
# 4) PA-TOA IMAGE ÜRETME
# ============================================================

def normalize_01(x):
    x = np.asarray(x, dtype=np.float32)

    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)

    mn = np.nanmin(x)
    mx = np.nanmax(x)

    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    return (x - mn) / (mx - mn)


def sequence_to_patoa_image(toa_values, amp_values, img_size=224):
    """
    TOA / TOAD + Amplitude dizisini 224x224 PA-TOA image'a çevirir.

    x ekseni: relative TOA / frame_time
    y ekseni: normalized amplitude
    """
    toa = np.asarray(toa_values, dtype=np.float32)
    amp = np.asarray(amp_values, dtype=np.float32)

    mask = np.isfinite(toa) & np.isfinite(amp)
    toa = toa[mask]
    amp = amp[mask]

    if len(toa) == 0:
        arr = np.zeros((img_size, img_size), dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB")

    order = np.argsort(toa)
    toa = toa[order]
    amp = amp[order]

    x_norm = normalize_01(toa)
    y_norm = normalize_01(amp)

    img = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)

    points = []

    for x, y in zip(x_norm, y_norm):
        px = int(round(x * (img_size - 1)))

        # image koordinatında y aşağı doğru arttığı için tersliyoruz
        py = int(round((1.0 - y) * (img_size - 1)))

        points.append((px, py))

    # çizgi çiz
    if len(points) >= 2:
        draw.line(points, fill=255, width=2)

    # noktaları belirginleştir
    r = 2
    for px, py in points:
        draw.ellipse((px - r, py - r, px + r, py + r), fill=255)

    return img.convert("RGB")


# ============================================================
# 5) DATASET: GT SCAN CLUSTER -> PA-TOA IMAGE
# ============================================================

class PAToAScanDataset(Dataset):
    def __init__(self, merged_df, class_to_idx=None):
        self.samples = []

        df = merged_df.dropna(
            subset=["gt_scan_cluster", "tarama_tipi", "amplitude"]
        ).copy()

        # No_Scanning gerçek scan type değil; dataset sample'ı üretme.
        before = len(df)
        df = df[~df["tarama_tipi"].apply(is_no_scanning_label)].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"[BILGI] Dataset oluşturulmadan {removed} adet No_Scanning train atıldı.")

        # Her gerçek tarama kümesi bir training sample olacak.
        for cluster_id, g in df.groupby("gt_scan_cluster"):
            g = g.sort_values(["scenario_id", "frame_time"]).copy()

            if len(g) < MIN_TRAINS_PER_SCAN:
                continue

            # label = tarama tipi
            label_str = str(g["tarama_tipi"].mode().iloc[0])

            # TOA için frame_time kullanıyoruz.
            # İstersen "toad" yapabilirsin ama scan pattern için frame_time daha mantıklı.
            toa_values = g["frame_time"].values.astype(np.float32)
            amp_values = g["amplitude"].values.astype(np.float32)

            self.samples.append({
                "cluster_id": cluster_id,
                "label_str": label_str,
                "toa": toa_values,
                "amp": amp_values,
                "n": len(g),
            })

        labels = sorted(set(s["label_str"] for s in self.samples))

        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(labels)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # class_to_idx içinde olmayan sample varsa at
        self.samples = [
            s for s in self.samples
            if s["label_str"] in self.class_to_idx
        ]

        print("\nDataset hazır.")
        print("Sample sayısı:", len(self.samples))
        print("Class mapping:", self.class_to_idx)

        counts = Counter([s["label_str"] for s in self.samples])
        print("Class counts:")
        for k, v in counts.items():
            print(" ", k, ":", v)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = sequence_to_patoa_image(s["toa"], s["amp"], IMG_SIZE)

        arr = np.asarray(img).astype(np.float32) / 255.0

        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))

        x = torch.tensor(arr, dtype=torch.float32)
        y = torch.tensor(self.class_to_idx[s["label_str"]], dtype=torch.long)

        return x, y


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    n = len(dataset)

    indices = list(range(n))
    random.Random(SEED).shuffle(indices)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ============================================================
# 6) RESNET BOTTLENECK MODEL
# ============================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()

        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(
            mid_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ScanTypeResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.in_channels = 64

        # 224x224 -> 112x112
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 112x112 -> 56x56
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ResNet-50 style: 3, 4, 6, 3
        self.layer1 = self._make_layer(mid_channels=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(mid_channels=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(mid_channels=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(mid_channels=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Paper figürde 1000 yazıyor olabilir, ama bizde class sayısı kadar.
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, mid_channels, blocks, stride):
        out_channels = mid_channels * Bottleneck.expansion

        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        layers = []

        layers.append(
            Bottleneck(
                in_channels=self.in_channels,
                mid_channels=mid_channels,
                stride=stride,
                downsample=downsample
            )
        )

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    in_channels=self.in_channels,
                    mid_channels=mid_channels,
                    stride=1,
                    downsample=None
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)       # 224 -> 112
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)     # 112 -> 56

        x = self.layer1(x)      # 56x56x256
        x = self.layer2(x)      # 28x28x512
        x = self.layer3(x)      # 14x14x1024
        x = self.layer4(x)      # 7x7x2048

        x = self.avgpool(x)     # 1x1x2048
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ============================================================
# 7) TRAIN / TEST
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)
            loss = criterion(out, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred = out.argmax(dim=1)

            total_loss += loss.item() * x.size(0)
            total_correct += (pred == y).sum().item()
            total_count += x.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)

    return avg_loss, avg_acc


def evaluate_predictions(model, loader, idx_to_class):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()

            all_pred.extend(pred.tolist())
            all_true.extend(y.numpy().tolist())

    print("\n========== TEST RESULTS ==========")

    n_class = len(idx_to_class)

    cm = np.zeros((n_class, n_class), dtype=int)

    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    print("\nConfusion matrix:")
    print(cm)

    print("\nPer-class accuracy:")
    for i in range(n_class):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total else 0.0
        print(f"{i} / {idx_to_class[i]}: {acc:.4f} ({correct}/{total})")

    overall = np.trace(cm) / max(cm.sum(), 1)
    print(f"\nOverall test accuracy: {overall:.4f}")

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.txt")
    np.savetxt(cm_path, cm, fmt="%d", delimiter="\t")
    print("Confusion matrix kaydedildi:", cm_path)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Device:", DEVICE)

    print("\nVeriler okunuyor...")
    train_val_df = build_synthetic_training_dataset()
    test_df = build_scenario_cluster_test_dataset()

    train_val_dataset = PAToAScanDataset(train_val_df)
    test_dataset = PAToAScanDataset(test_df, class_to_idx=train_val_dataset.class_to_idx)

    if len(train_val_dataset) < 10:
        raise RuntimeError("Dataset çok küçük görünüyor. Kolon/path eşleşmesini kontrol et.")

    if len(test_dataset) < 1:
        raise RuntimeError("Scenario test dataset bos gorunuyor. Cluster output pathlerini kontrol et.")

    num_classes = len(train_val_dataset.class_to_idx)

    train_set, val_set, _ = split_dataset(train_val_dataset, train_ratio=0.8, val_ratio=0.2)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = ScanTypeResNet(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = -1.0
    best_path = os.path.join(OUTPUT_DIR, "best_patoa_resnet.pt")

    print("\nTraining başlıyor...")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_val_dataset.class_to_idx,
                    "idx_to_class": train_val_dataset.idx_to_class,
                    "img_size": IMG_SIZE,
                },
                best_path
            )
            print("  En iyi model kaydedildi:", best_path)

    print("\nBest val acc:", best_val_acc)

    print("\nEn iyi model yükleniyor...")
    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_predictions(model, test_loader, train_val_dataset.idx_to_class)

    print("\nBitti.")


if __name__ == "__main__":
    main()
