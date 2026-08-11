import os
import re
import glob
import math
import numpy as np
import pandas as pd


# ============================================================
# 1) PATH AYARLARI
# ============================================================

FEATURE_DIR = r"C:\Users\Bogazici\Desktop\senaryolar\senaryo 9\Darbe Dizi özellikleri"

SCAN_DIR = r"C:\Users\Bogazici\Desktop\senaryolar\senaryo 9\antenna_scanning"

SCANNING_EMITTER_IDS = [0, 2, 4, 5, 6, 8]


# ============================================================
# 2) CONFIG
# ============================================================

CONFIG = {
    "assign_threshold": 0.60,
    "default_expected_dt": 5.0,

    "pw_tol": 1.0,
    "fc_tol": 10.0,
    "bw_tol": 1.0,
    "toad_tol": 1000.0,
    "amp_tol": 1.0,

    "long_gap_factor": 5.0,
    "fade_drop_threshold": 0.8,
    "association_recent_train_count": 5,

}


# ============================================================
# 3) YARDIMCI FONKSİYONLAR
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


def numeric_similarity(a, b, tol):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    d = abs(float(a) - float(b))
    return math.exp(-d / max(tol, 1e-9))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ============================================================
# 4) DARBE DİZİ ÖZELLİKLERİ DOSYALARINI OKU
# ============================================================

def read_single_feature_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        return None

    header_line = lines[0].replace("#", "").strip()
    value_line = lines[1].strip()

    header = split_whitespace_line(header_line)
    values = split_whitespace_line(value_line)

    if len(values) != len(header):
        return None

    data = dict(zip(header, values))

    filename = os.path.basename(file_path)
    frame_time, file_freq_mhz = parse_frame_time_and_freq(filename)

    row = {
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

    return row


def read_all_feature_files(feature_dir):
    files = sorted(glob.glob(os.path.join(feature_dir, "*.txt")))
    rows = []

    for fp in files:
        row = read_single_feature_file(fp)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Darbe Dizi özellikleri klasöründen hiç veri okunamadı.")

    df = df.sort_values(["frame_time", "emitter_id", "fc", "pulse_train_id"], na_position="last")
    df = df.reset_index(drop=True)
    return df


# ============================================================
# 5) ANTENNA_SCANNING DOSYALARINI OKU
# ============================================================

def read_single_scanning_file(file_path, emitter_id):
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

    keep_original = [c for c in col_map.keys() if c in df.columns]
    keep_original.append("emitter_id")

    df = df[keep_original].rename(columns=col_map)

    df["emitter_id"] = df["emitter_id"].apply(safe_int)
    df["pulse_train_id"] = df["pulse_train_id"].apply(safe_int)

    if "tarama_idx" in df.columns:
        df["tarama_idx"] = df["tarama_idx"].apply(safe_int)
    else:
        df["tarama_idx"] = np.nan

    if "amplitude" in df.columns:
        df["amplitude"] = df["amplitude"].apply(safe_float)
    else:
        df["amplitude"] = np.nan

    return df


def read_all_scanning_files(scan_dir):
    rows = []

    for emitter_id in SCANNING_EMITTER_IDS:
        fp = os.path.join(scan_dir, f"emitter_{emitter_id}.txt")
        if not os.path.exists(fp):
            continue

        df_e = read_single_scanning_file(fp, emitter_id)
        if not df_e.empty:
            rows.append(df_e)

    if len(rows) == 0:
        raise RuntimeError("antenna_scanning klasöründen hiç veri okunamadı.")

    scan_df = pd.concat(rows, ignore_index=True)
    scan_df = scan_df.drop_duplicates(subset=["emitter_id", "pulse_train_id"], keep="first")
    scan_df = scan_df.reset_index(drop=True)
    return scan_df


def build_merged_dataset(feature_dir, scan_dir):
    feature_df = read_all_feature_files(feature_dir)
    scan_df = read_all_scanning_files(scan_dir)

    merged = feature_df.merge(
        scan_df,
        on=["emitter_id", "pulse_train_id"],
        how="left",
        validate="many_to_one"
    )

    merged["gt_cluster"] = np.where(
        merged["tarama_idx"].notna(),
        "emitter_" + merged["emitter_id"].astype("Int64").astype(str)
        + "_scan_" + merged["tarama_idx"].astype("Int64").astype(str),
        np.nan
    )

    merged = merged.sort_values(["frame_time", "emitter_id", "fc", "pulse_train_id"], na_position="last")
    merged = merged.reset_index(drop=True)
    return merged


# ============================================================
# 6) SON N TRAIN'I CLUSTER GİBİ TUTAN CLASS
# ============================================================

class ManualRecentCluster:
    def __init__(self, rows):
        """
        rows: cluster gibi kullanılacak train satırları.
        Ana koddaki recent centroid mantığını taklit eder.
        """
        rows = rows.sort_values("frame_time").copy()

        self.rows = rows
        self.members = list(rows.index.astype(int))

        last_row = rows.iloc[-1]

        self.emitter_id = last_row["emitter_id"]
        self.operation_mode_id = last_row["operation_mode_id"]
        self.modulation = last_row["modulation"]
        self.pri_mod_type = last_row["pri_mod_type"]

        self.frame_times = [
            float(x) if not pd.isna(x) else np.nan
            for x in rows["frame_time"].values
        ]

        self.amplitudes = [
            float(x) if not pd.isna(x) else np.nan
            for x in rows["amplitude"].values
        ]

        valid_times = [x for x in self.frame_times if not pd.isna(x)]
        self.last_time = max(valid_times) if valid_times else None

        # Bu iki sayaç ana koddaki circular_probability için gerekli.
        self.long_gap_count = 0
        self.fade_before_long_gap_count = 0

        self._estimate_gap_counters_from_given_rows()

    def _estimate_gap_counters_from_given_rows(self):
        """
        Ana kodda long_gap_count ve fade_before_long_gap_count cluster'a train eklenirken oluşuyor.
        Burada elimizde verilen 5 train olduğu için bunlardan yaklaşık olarak hesaplıyoruz.
        """
        if len(self.frame_times) < 2:
            return

        expected_dt = self.expected_dt()
        if expected_dt <= 0:
            expected_dt = CONFIG["default_expected_dt"]

        for i in range(1, len(self.frame_times)):
            t_prev = self.frame_times[i - 1]
            t_now = self.frame_times[i]

            if pd.isna(t_prev) or pd.isna(t_now):
                continue

            gap = float(t_now) - float(t_prev)
            gap_factor = gap / expected_dt

            if gap_factor >= CONFIG["long_gap_factor"]:
                self.long_gap_count += 1

                # Gap öncesi amplitude düşmüş mü?
                # Ana koddaki mantığa yakın olsun diye gap'ten önceki son amplitude'a bakıyoruz.
                if i >= 1:
                    prev_amp = self.amplitudes[i - 1]
                    recent_window = self.amplitudes[max(0, i - 4):i]

                    if len(recent_window) > 0:
                        recent_max = np.nanmax(recent_window)

                        if not pd.isna(recent_max) and not pd.isna(prev_amp):
                            if recent_max - prev_amp >= CONFIG["fade_drop_threshold"]:
                                self.fade_before_long_gap_count += 1

    def expected_dt(self):
        ts = np.array([x for x in self.frame_times if not pd.isna(x)], dtype=float)

        if len(ts) < 2:
            return CONFIG["default_expected_dt"]

        diffs = np.diff(np.sort(ts))
        diffs = diffs[diffs > 0]

        if len(diffs) == 0:
            return CONFIG["default_expected_dt"]

        return float(np.median(diffs))

    def lifetime(self):
        ts = [x for x in self.frame_times if not pd.isna(x)]

        if len(ts) < 2:
            return CONFIG["default_expected_dt"]

        return max(ts) - min(ts) + CONFIG["default_expected_dt"]

    def gap_ratio_proxy(self):
        expected_dt = self.expected_dt()

        if expected_dt <= 0:
            expected_dt = CONFIG["default_expected_dt"]

        ts = np.array([x for x in self.frame_times if not pd.isna(x)], dtype=float)

        if len(ts) < 2:
            return 0.0

        diffs = np.diff(np.sort(ts))
        missing = np.sum(np.maximum(0.0, diffs - expected_dt))

        return float(missing / max(self.lifetime(), 1e-9))

    def circular_probability(self):
        gap_ratio = self.gap_ratio_proxy()

        x = (
            1.1 * self.long_gap_count
            + 1.5 * self.fade_before_long_gap_count
            + 3.0 * gap_ratio
            - 2.2
        )

        return sigmoid(x)

    def numeric_centroid(self, col):
        n_recent = CONFIG["association_recent_train_count"]
        vals = self.rows.tail(n_recent)[col].dropna().values

        if len(vals) == 0:
            return np.nan

        return float(np.mean(vals))

# ============================================================
# 7) BİR TRAIN'I, VERİLEN 5 TRAIN CLUSTER'INA KARŞI SORGULA
# ============================================================

def get_single_train_row(df, train_id, emitter_id=None):
    train_id = int(train_id)
    q = df[df["pulse_train_id"] == train_id].copy()

    if emitter_id is not None and str(emitter_id).strip() != "":
        emitter_id = int(emitter_id)
        q = q[q["emitter_id"] == emitter_id].copy()

    if q.empty:
        raise ValueError(f"Train bulunamadı: pulse_train_id={train_id}, emitter_id={emitter_id}")

    if len(q) > 1:
        print("\n[UYARI] Bu pulse_train_id birden fazla satırda bulundu.")
        print("Daha net sonuç için emitter_id gir. İlk satır kullanılacak.")
        print(
            q[
                [
                    "emitter_id",
                    "pulse_train_id",
                    "frame_time",
                    "fc",
                    "operation_mode_id",
                    "modulation",
                    "pri_mod_type",
                ]
            ].to_string(index=False)
        )

    return q.iloc[0]


def get_multiple_train_rows(df, train_ids, emitter_id=None):
    train_ids = [int(x) for x in train_ids]

    rows = []

    for tid in train_ids:
        row = get_single_train_row(df, tid, emitter_id)
        rows.append(row)

    cluster_rows = pd.DataFrame(rows)

    # Orijinal df indexlerini korumak için tekrar seçelim.
    idxs = [row.name for row in rows]
    cluster_rows = df.loc[idxs].copy()

    cluster_rows = cluster_rows.sort_values("frame_time")

    return cluster_rows


def association_score_components_for_manual_cluster(new_row, cluster_rows):
    cluster = ManualRecentCluster(cluster_rows)

    pw_c = cluster.numeric_centroid("pw")
    fc_c = cluster.numeric_centroid("fc")
    bw_c = cluster.numeric_centroid("bw")
    toad_c = cluster.numeric_centroid("toad")
    amp_c = cluster.numeric_centroid("amplitude")

    pw_s = numeric_similarity(new_row["pw"], pw_c, CONFIG["pw_tol"])
    fc_s = numeric_similarity(new_row["fc"], fc_c, CONFIG["fc_tol"])
    bw_s = numeric_similarity(new_row["bw"], bw_c, CONFIG["bw_tol"])
    toad_s = numeric_similarity(new_row["toad"], toad_c, CONFIG["toad_tol"])
    amp_s = numeric_similarity(new_row["amplitude"], amp_c, CONFIG["amp_tol"])

    expected_dt = cluster.expected_dt()
    p_circ = cluster.circular_probability()
    effective_expected = expected_dt * (1.0 + 4.0 * p_circ)

    if cluster.last_time is None or pd.isna(new_row["frame_time"]):
        gap = np.nan
        time_s = 0.5
    else:
        gap = float(new_row["frame_time"]) - float(cluster.last_time)
        time_s = math.exp(
            -max(0.0, gap - expected_dt) / max(effective_expected, 1e-9)
        )

    score = (
        0.22 * pw_s
        + 0.22 * fc_s
        + 0.16 * bw_s
        + 0.14 * toad_s
        + 0.10 * amp_s
        + 0.16 * time_s
    )

    return {
        "score": float(score),
        "passed_threshold": bool(score >= CONFIG["assign_threshold"]),
        "threshold": CONFIG["assign_threshold"],
        "pw_s": float(pw_s),
        "fc_s": float(fc_s),
        "bw_s": float(bw_s),
        "toad_s": float(toad_s),
        "amp_s": float(amp_s),
        "time_s": float(time_s),
        "expected_dt": float(expected_dt),
        "gap": float(gap) if not pd.isna(gap) else np.nan,
        "p_circ": float(p_circ),
        "effective_expected": float(effective_expected),
        "long_gap_count": cluster.long_gap_count,
        "fade_before_long_gap_count": cluster.fade_before_long_gap_count,
        "gap_ratio": float(cluster.gap_ratio_proxy()),
        "centroids": {
            "pw": pw_c,
            "fc": fc_c,
            "bw": bw_c,
            "toad": toad_c,
            "amplitude": amp_c,
        },
    }


def parse_train_id_list(s):
    s = s.replace(" ", "")
    parts = [x for x in s.split(",") if x != ""]

    if len(parts) == 0:
        raise ValueError("Hiç train ID girilmedi.")

    return [int(x) for x in parts]


def print_manual_cluster_association_report(df, new_train_id, cluster_train_ids, new_emitter_id=None, cluster_emitter_id=None):
    new_row = get_single_train_row(df, new_train_id, new_emitter_id)
    cluster_rows = get_multiple_train_rows(df, cluster_train_ids, cluster_emitter_id)

    result = association_score_components_for_manual_cluster(new_row, cluster_rows)

    print("\n========== MANUAL 5-TRAIN ASSOCIATION CHECK ==========")

    print("\nYeni gelen train:")
    print(
        new_row[
            [
                "emitter_id",
                "pulse_train_id",
                "frame_time",
                "pw",
                "fc",
                "bw",
                "toad",
                "amplitude",
                "operation_mode_id",
                "modulation",
                "pri_mod_type",
                "gt_cluster",
            ]
        ].to_string()
    )

    print("\nCluster gibi kullanılan trainler:")
    print(
        cluster_rows[
            [
                "emitter_id",
                "pulse_train_id",
                "frame_time",
                "pw",
                "fc",
                "bw",
                "toad",
                "amplitude",
                "operation_mode_id",
                "modulation",
                "pri_mod_type",
                "gt_cluster",
            ]
        ].to_string(index=False)
    )

    print(f"\nThreshold: {result['threshold']:.4f}")

    print("\nCentroid değerleri, verilen cluster trainlerinden hesaplandı:")
    for k, v in result["centroids"].items():
        print(f"  {k}: {v}")

    print("\nTime bilgisi:")
    print(f"  cluster last_time: {cluster_rows['frame_time'].max()}")
    print(f"  new train time:    {new_row['frame_time']}")
    print(f"  gap:               {result['gap']}")
    print(f"  expected_dt:       {result['expected_dt']}")
    print(f"  p_circ:            {result['p_circ']:.4f}")
    print(f"  effective_expected:{result['effective_expected']:.4f}")
    print(f"  long_gap_count:    {result['long_gap_count']}")
    print(f"  fade_gap_count:    {result['fade_before_long_gap_count']}")
    print(f"  gap_ratio:         {result['gap_ratio']:.4f}")

    print("\nBileşen skorları:")
    print(f"  pw_s:   {result['pw_s']:.4f}   ağırlık=0.22   katkı={0.22 * result['pw_s']:.4f}")
    print(f"  fc_s:   {result['fc_s']:.4f}   ağırlık=0.22   katkı={0.22 * result['fc_s']:.4f}")
    print(f"  bw_s:   {result['bw_s']:.4f}   ağırlık=0.16   katkı={0.16 * result['bw_s']:.4f}")
    print(f"  toad_s: {result['toad_s']:.4f}   ağırlık=0.14   katkı={0.14 * result['toad_s']:.4f}")
    print(f"  amp_s:  {result['amp_s']:.4f}   ağırlık=0.10   katkı={0.10 * result['amp_s']:.4f}")
    print(f"  time_s: {result['time_s']:.4f}   ağırlık=0.16   katkı={0.16 * result['time_s']:.4f}")

    print(f"\nFinal score: {result['score']:.4f}")

    if result["passed_threshold"]:
        print("SONUÇ: YÜKSEK -> threshold üstü, ana koda göre bu cluster'a atanır.")
    else:
        print("SONUÇ: DÜŞÜK -> threshold altı, ana koda göre yeni cluster açılır.")

    print("\nYeni train ile centroid farkları:")
    for col in ["pw", "fc", "bw", "toad", "amplitude"]:
        centroid_val = result["centroids"][col]
        new_val = new_row[col]
        diff = abs(float(new_val) - float(centroid_val)) if not pd.isna(new_val) and not pd.isna(centroid_val) else np.nan
        print(f"  {col}: new={new_val}, centroid={centroid_val}, abs_diff={diff}")


# ============================================================
# 8) MAIN
# ============================================================

def main():
    merged_df = build_merged_dataset(FEATURE_DIR, SCAN_DIR)

    new_train_id = input("Yeni gelen train ID gir: ").strip()
    new_emitter_id = input("Yeni gelen train emitter_id opsiyonel, boş geçebilirsin: ").strip()

    cluster_train_ids_str = input("Cluster gibi kullanılacak son train ID'leri virgülle gir, örn 1,6,11,16,21: ").strip()
    cluster_emitter_id = input("Cluster trainleri için emitter_id opsiyonel, boş geçebilirsin: ").strip()

    new_emitter_id = None if new_emitter_id == "" else new_emitter_id
    cluster_emitter_id = None if cluster_emitter_id == "" else cluster_emitter_id

    cluster_train_ids = parse_train_id_list(cluster_train_ids_str)

    print_manual_cluster_association_report(
        merged_df,
        new_train_id=new_train_id,
        cluster_train_ids=cluster_train_ids,
        new_emitter_id=new_emitter_id,
        cluster_emitter_id=cluster_emitter_id,
    )


if __name__ == "__main__":
    main()
