import os
import re
import glob
import math
import numpy as np
import pandas as pd


# ============================================================
# 1) PATH AYARLARI
# ============================================================

FEATURE_DIR = r"C:\Users\Bogazici\Desktop\senaryolar\senaryo 5\Darbe Dizi özellikleri"

SCAN_DIR = r"C:\Users\Bogazici\Desktop\senaryolar\senaryo 5\antenna_scanning"

OUTPUT_DIR = r"C:\Users\Bogazici\Desktop\senaryolar\scan_tour_clustering_outputs5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SCANNING_EMITTER_IDS = [0,5,8]


# ============================================================
# 2) CONFIG
# ============================================================

CONFIG = {
    # Association threshold: yeni pulse train mevcut cluster'a atanacak mÄ±?
    "assign_threshold": 0.60,

    # Time gap normalizasyonu.
    # frame_time farklarÄ± bÃ¼yÃ¼kse bunu dataset'e gÃ¶re deÄŸiÅŸtir.
    "default_expected_dt": 5.0,

    # Cluster kaÃ§ frame/time gÃ¶rÃ¼nmezse hidden/dead olsun?
    # Circular-like clusterlarda bu otomatik uzatÄ±lÄ±yor.
    "base_death_timeout": 150.0,

    # SayÄ±sal feature toleranslarÄ±.
    # AynÄ± scanning tour iÃ§in bu parametreler deÄŸiÅŸmiyor demiÅŸtin.
    # Ama float farklarÄ± olabilir diye kÃ¼Ã§Ã¼k tolerans koyuyoruz.
    "pw_tol": 1.0,
    "fc_tol": 10.0,
    "bw_tol": 1.0,
    "toad_tol": 1000.0,

    # Amplitude scoring.
    "amp_tol": 1.0,

    # Association score centroid uses only the most recent N trains in a cluster.
    # If the cluster has fewer than N trains, all available trains are used.
    "association_recent_train_count": 5,

    # Circular-like gap detection.
    "long_gap_factor": 5.0,
    "fade_drop_threshold": 0.8,
}


# ============================================================
# 3) YARDIMCI FONKSÄ°YONLAR
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
    Ã–rnek:
        0.0db_1200_1230MHz(1).txt

    Ã‡Ä±ktÄ±:
        frame_time = 1200
        file_freq_mhz = 1230
    """
    base = os.path.basename(filename)

    m = re.search(r"db_(\d+)_([0-9.]+)\s*MHz", base, flags=re.IGNORECASE)
    if m:
        frame_time = int(m.group(1))
        file_freq_mhz = float(m.group(2))
        return frame_time, file_freq_mhz

    # Daha gevÅŸek fallback:
    # ilk underscore'dan sonraki sayÄ± frame olsun.
    m2 = re.search(r"_(\d+)_", base)
    frame_time = int(m2.group(1)) if m2 else np.nan

    m3 = re.search(r"_([0-9.]+)\s*MHz", base, flags=re.IGNORECASE)
    file_freq_mhz = float(m3.group(1)) if m3 else np.nan

    return frame_time, file_freq_mhz


def split_whitespace_line(line):
    return line.strip().split()


def split_table_line_2plus_spaces(line):
    """
    antenna_scanning dosyasÄ±nda kolon isimleri boÅŸluklu:
        Tarama Tipi        Tarama idx  AdÄ±m (Â°) ...
    Bu yÃ¼zden 2 veya daha fazla boÅŸluÄŸa gÃ¶re split ediyoruz.
    """
    return re.split(r"\s{2,}", line.strip())


def normalize_mode(mode_str):
    """
    mode_1 -> 1
    mode_2 -> 2
    """
    if pd.isna(mode_str):
        return np.nan
    m = re.search(r"mode[_\s-]*(\d+)", str(mode_str), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return safe_int(mode_str)


def numeric_similarity(a, b, tol):
    """
    1'e yakÄ±nsa Ã§ok benzer, 0'a yakÄ±nsa uzak.
    """
    if pd.isna(a) or pd.isna(b):
        return 0.0
    d = abs(float(a) - float(b))
    return math.exp(-d / max(tol, 1e-9))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def mode_or_nan(series):
    vals = series.dropna().values
    if len(vals) == 0:
        return np.nan
    return pd.Series(vals).mode().iloc[0]


# ============================================================
# 4) DARBE DÄ°ZÄ° Ã–ZELLÄ°KLERÄ° DOSYALARINI OKU
# ============================================================

def read_single_feature_file(file_path):
    """
    Dosya formatÄ±:
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

    if len(values) != len(header):
        print(f"[UYARI] Kolon sayÄ±sÄ± uyuÅŸmadÄ±, geÃ§iliyor: {os.path.basename(file_path)}")
        print("Header count:", len(header), "Value count:", len(values))
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

        # Clustering inputuna antenna_scan_type koymayacaÄŸÄ±z.
        # Ama istersek debug/evaluation iÃ§in saklayabiliriz.
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
        raise RuntimeError("Darbe Dizi Ã¶zellikleri klasÃ¶rÃ¼nden hiÃ§ veri okunamadÄ±.")

    df = df.sort_values(["frame_time", "emitter_id", "fc", "pulse_train_id"], na_position="last")
    df = df.reset_index(drop=True)

    return df


# ============================================================
# 5) ANTENNA_SCANNING DOSYALARINI OKU
# ============================================================

def read_single_scanning_file(file_path, emitter_id):
    """
    Dosya formatÄ±:
        Tarama Tipi        Tarama idx  AdÄ±m (Â°)  Darbe Dizi idx  ... Amplitude ...
        -----------------  ----------  --------  --------------  ...
        raster_scanning    3           3.96      1               ... 2.0157 ...
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]

    if len(lines) < 3:
        return pd.DataFrame()

    header = split_table_line_2plus_spaces(lines[0])
    data_lines = lines[2:]  # 1. satÄ±r header, 2. satÄ±r dashed separator

    rows = []
    for ln in data_lines:
        parts = split_table_line_2plus_spaces(ln)

        if len(parts) != len(header):
            # BazÄ± satÄ±rlar bozuk/truncated olabilir. GeÃ§iyoruz.
            continue

        row = dict(zip(header, parts))
        row["emitter_id"] = emitter_id
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Sadece lazÄ±m olanlarÄ± alÄ±yoruz:
    # - amplitude input feature
    # - tarama_idx ground truth
    # - tarama_tipi debug/evaluation iÃ§in saklanabilir, modele verilmeyecek
    col_map = {
        "Tarama Tipi": "tarama_tipi",
        "Tarama idx": "tarama_idx",
        "Darbe Dizi idx": "pulse_train_id",
        "Amplitude": "amplitude",
    }

    missing = [c for c in col_map.keys() if c not in df.columns]
    if missing:
        print(f"[UYARI] {os.path.basename(file_path)} iÃ§inde eksik kolonlar:", missing)

    keep_original = [c for c in col_map.keys() if c in df.columns]
    keep_original.append("emitter_id")

    df = df[keep_original].rename(columns=col_map)

    df["emitter_id"] = df["emitter_id"].apply(safe_int)
    df["pulse_train_id"] = df["pulse_train_id"].apply(safe_int)
    df["tarama_idx"] = df["tarama_idx"].apply(safe_int)
    df["amplitude"] = df["amplitude"].apply(safe_float)

    return df


def read_all_scanning_files(scan_dir):
    rows = []

    for emitter_id in SCANNING_EMITTER_IDS:
        fp = os.path.join(scan_dir, f"emitter_{emitter_id}.txt")

        if not os.path.exists(fp):
            print(f"[UYARI] BulunamadÄ±: {fp}")
            continue

        df_e = read_single_scanning_file(fp, emitter_id)
        if not df_e.empty:
            rows.append(df_e)

    if len(rows) == 0:
        raise RuntimeError("antenna_scanning klasÃ¶rÃ¼nden hiÃ§ veri okunamadÄ±.")

    scan_df = pd.concat(rows, ignore_index=True)

    # AynÄ± emitter_id + pulse_train_id birden fazla geldiyse ilkini al.
    # Normalde tekil olmalÄ±.
    scan_df = scan_df.drop_duplicates(subset=["emitter_id", "pulse_train_id"], keep="first")
    scan_df = scan_df.reset_index(drop=True)

    return scan_df


# ============================================================
# 6) MERGE: FEATURE + AMPLITUDE + GROUND TRUTH
# ============================================================

def build_merged_dataset(feature_dir, scan_dir):
    feature_df = read_all_feature_files(feature_dir)
    scan_df = read_all_scanning_files(scan_dir)

    merged = feature_df.merge(
        scan_df,
        on=["emitter_id", "pulse_train_id"],
        how="left",
        validate="many_to_one"
    )

    # Composite ground truth:
    # farklÄ± emitter dosyalarÄ±nda aynÄ± tarama_idx olabildiÄŸi iÃ§in
    # sadece tarama_idx kullanmak yanlÄ±ÅŸ.
    merged["gt_cluster"] = np.where(
        merged["tarama_idx"].notna(),
        "emitter_" + merged["emitter_id"].astype("Int64").astype(str)
        + "_scan_" + merged["tarama_idx"].astype("Int64").astype(str),
        np.nan
    )

    # Model inputuna girmeyecek kolonlar:
    # antenna_scan_type_raw, tarama_tipi, tarama_idx, gt_cluster
    merged = merged.sort_values(["frame_time", "emitter_id", "fc", "pulse_train_id"], na_position="last")
    merged = merged.reset_index(drop=True)

    return merged


# ============================================================
# 7) ONLINE BIRTH-DEATH CLUSTERING
# ============================================================

class OnlineCluster:
    def __init__(self, cluster_id, row):
        self.cluster_id = cluster_id

        self.emitter_id = row["emitter_id"]
        self.operation_mode_id = row["operation_mode_id"]
        self.modulation = row["modulation"]
        self.pri_mod_type = row["pri_mod_type"]

        self.members = []
        self.frame_times = []
        self.amplitudes = []

        self.last_time = None
        self.first_time = None

        self.long_gap_count = 0
        self.fade_before_long_gap_count = 0

        self.is_dead = False

        self.add(row)

    def add(self, row):
        t = row["frame_time"]
        a = row["amplitude"]

        if self.last_time is not None and not pd.isna(t):
            gap = float(t) - float(self.last_time)
            expected_dt = self.expected_dt()

            if expected_dt <= 0:
                expected_dt = CONFIG["default_expected_dt"]

            gap_factor = gap / expected_dt

            if gap_factor >= CONFIG["long_gap_factor"]:
                self.long_gap_count += 1

                # Gap Ã¶ncesi amplitude dÃ¼ÅŸÃ¼ÅŸÃ¼ var mÄ±?
                if len(self.amplitudes) >= 2:
                    recent_max = np.nanmax(self.amplitudes[-4:])
                    prev_amp = self.amplitudes[-1]
                    if not pd.isna(recent_max) and not pd.isna(prev_amp):
                        if recent_max - prev_amp >= CONFIG["fade_drop_threshold"]:
                            self.fade_before_long_gap_count += 1

        self.members.append(int(row.name))
        self.frame_times.append(float(t) if not pd.isna(t) else np.nan)
        self.amplitudes.append(float(a) if not pd.isna(a) else np.nan)

        if self.first_time is None and not pd.isna(t):
            self.first_time = float(t)

        if not pd.isna(t):
            self.last_time = float(t)

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
        """
        Circular diye hard label kullanmÄ±yoruz.
        Sadece cluster'Ä±n Ã§ok kesilmeli davranÄ±p davranmadÄ±ÄŸÄ±nÄ± tahmin ediyoruz.
        """
        gap_ratio = self.gap_ratio_proxy()

        x = (
            1.1 * self.long_gap_count
            + 1.5 * self.fade_before_long_gap_count
            + 3.0 * gap_ratio
            - 2.2
        )

        return sigmoid(x)

    def death_timeout(self):
        """
        Circular-like clusterlar daha uzun hidden kalabilsin.
        """
        p_circ = self.circular_probability()
        return CONFIG["base_death_timeout"] * (1.0 + 3.0 * p_circ)

    def numeric_centroid(self, df, col):
        vals = df.loc[self.members, col].dropna().values
        if len(vals) == 0:
            return np.nan
        return float(np.mean(vals))

    def recent_numeric_centroid(self, df, col):
        n_recent = CONFIG["association_recent_train_count"]
        recent_members = self.members[-n_recent:]
        vals = df.loc[recent_members, col].dropna().values
        if len(vals) == 0:
            return np.nan
        return float(np.mean(vals))

    def association_score(self, row, df):
        """
        Model input:
            frame_time, emitter_id, operation_mode_id, modulation,
            pri_mod_type, TOAD, PW, Fc, BW, Amplitude

        Ground truth yok.
        antenna_scan_type yok.
        tarama_idx yok.
        """
        if self.is_dead:
            return -1.0

        pw_c = self.recent_numeric_centroid(df, "pw")
        fc_c = self.recent_numeric_centroid(df, "fc")
        bw_c = self.recent_numeric_centroid(df, "bw")
        toad_c = self.recent_numeric_centroid(df, "toad")
        amp_c = self.recent_numeric_centroid(df, "amplitude")

        pw_s = numeric_similarity(row["pw"], pw_c, CONFIG["pw_tol"])
        fc_s = numeric_similarity(row["fc"], fc_c, CONFIG["fc_tol"])
        bw_s = numeric_similarity(row["bw"], bw_c, CONFIG["bw_tol"])
        toad_s = numeric_similarity(row["toad"], toad_c, CONFIG["toad_tol"])

        # Amplitude deÄŸiÅŸebilir; Ã§ok sert ceza vermiyoruz.
        amp_s = numeric_similarity(row["amplitude"], amp_c, CONFIG["amp_tol"])

        # Time score
        if self.last_time is None or pd.isna(row["frame_time"]):
            time_s = 0.5
        else:
            gap = float(row["frame_time"]) - float(self.last_time)
            expected_dt = self.expected_dt()

            if expected_dt <= 0:
                expected_dt = CONFIG["default_expected_dt"]

            # Gap bÃ¼yÃ¼dÃ¼kÃ§e dÃ¼ÅŸer ama circular-like clusterlarda Ã§ok dÃ¼ÅŸmesin.
            p_circ = self.circular_probability()
            effective_expected = expected_dt * (1.0 + 4.0 * p_circ)
            time_s = math.exp(-max(0.0, gap - expected_dt) / max(effective_expected, 1e-9))

        # Weighted score.
        score = (
            0.22 * pw_s
            + 0.22 * fc_s
            + 0.16 * bw_s
            + 0.14 * toad_s
            + 0.10 * amp_s
            + 0.16 * time_s
        )

        return float(score)


def online_birth_death_clustering(df):
    df = df.copy()

    # Clustering inputunda eksik olanlar varsa doldurmadan kullanÄ±yoruz.
    df = df.sort_values(["frame_time", "emitter_id", "fc", "pulse_train_id"], na_position="last")
    df = df.reset_index(drop=True)

    clusters = []
    assignments = []

    next_cluster_id = 0

    for idx, row in df.iterrows():
        current_time = row["frame_time"]

        # Ã–nce Ã§ok eski clusterlarÄ± dead yap.
        for c in clusters:
            if c.is_dead:
                continue
            if c.last_time is None or pd.isna(current_time):
                continue

            if float(current_time) - float(c.last_time) > c.death_timeout():
                c.is_dead = True

        # Aday cluster skorlarÄ±
        best_cluster = None
        best_score = -1.0

        for c in clusters:
            score = c.association_score(row, df)
            if score > best_score:
                best_score = score
                best_cluster = c

        # Atama veya yeni birth
        if best_cluster is not None and best_score >= CONFIG["assign_threshold"]:
            best_cluster.add(row)
            assigned_id = best_cluster.cluster_id
        else:
            new_cluster = OnlineCluster(next_cluster_id, row)
            clusters.append(new_cluster)
            assigned_id = next_cluster_id
            next_cluster_id += 1

        assignments.append(assigned_id)

    df["pred_cluster"] = assignments

    # Her cluster iÃ§in circular probability yaz.
    circ_prob_map = {}
    for c in clusters:
        circ_prob_map[c.cluster_id] = c.circular_probability()

    df["pred_cluster_circular_prob"] = df["pred_cluster"].map(circ_prob_map)

    return df, clusters


# ============================================================
# 8) EVALUATION
# ============================================================

def pairwise_f1_score(y_true, y_pred):
    """
    sklearn yoksa fallback.
    AynÄ± cluster mÄ± deÄŸil mi pairwise bakar.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)
    tp = fp = fn = 0

    for i in range(n):
        for j in range(i + 1, n):
            same_true = y_true[i] == y_true[j]
            same_pred = y_pred[i] == y_pred[j]

            if same_true and same_pred:
                tp += 1
            elif (not same_true) and same_pred:
                fp += 1
            elif same_true and (not same_pred):
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1


def evaluate_clustering(df):
    eval_df = df.dropna(subset=["gt_cluster", "pred_cluster"]).copy()

    if eval_df.empty:
        print("[UYARI] Evaluation iÃ§in gt_cluster yok.")
        return

    y_true = eval_df["gt_cluster"].astype(str).values
    y_pred = eval_df["pred_cluster"].astype(str).values

    print("\n========== EVALUATION ==========")
    print("Evaluated pulse trains:", len(eval_df))
    print("GT cluster count:", eval_df["gt_cluster"].nunique())
    print("Pred cluster count:", eval_df["pred_cluster"].nunique())

    # Pairwise precision/recall/F1 her zaman hesaplansÄ±n.
    # Bu metrik aynÄ± GT cluster mÄ± / aynÄ± pred cluster mÄ± diye tÃ¼m Ã§iftlere bakÄ±yor.
    precision, recall, f1 = pairwise_f1_score(y_true, y_pred)
    print(f"Pairwise precision: {precision:.4f}")
    print(f"Pairwise recall:    {recall:.4f}")
    print(f"Pairwise F1:        {f1:.4f}")

    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

        print(f"ARI: {ari:.4f}")
        print(f"NMI: {nmi:.4f}")

    except Exception:
        print("sklearn yok, sadece pairwise precision/recall/F1 hesaplandÄ±.")

    # Cluster-level Ã¶zet
    summary = (
        eval_df
        .groupby("pred_cluster")
        .agg(
            n=("pred_cluster", "size"),
            gt_majority=("gt_cluster", mode_or_nan),
            emitter_majority=("emitter_id", mode_or_nan),
            t_min=("frame_time", "min"),
            t_max=("frame_time", "max"),
            amp_min=("amplitude", "min"),
            amp_max=("amplitude", "max"),
            circular_prob=("pred_cluster_circular_prob", "mean"),
        )
        .reset_index()
    )

    summary["amp_range"] = summary["amp_max"] - summary["amp_min"]

    summary_path = os.path.join(OUTPUT_DIR, "cluster_summary.txt")
    summary.to_csv(summary_path, index=False, sep="\t", encoding="utf-8-sig")

    print("Cluster summary kaydedildi:", summary_path)

    # GT-first recovery report:
    # For each antenna_scanning cluster, find the predicted cluster that captured
    # the largest percentage of its pulse trains.
    recovery_rows = []
    for gt_cluster, gt_part in eval_df.groupby("gt_cluster"):
        total = len(gt_part)
        pred_counts = gt_part["pred_cluster"].value_counts()
        best_pred_cluster = pred_counts.index[0]
        captured = int(pred_counts.iloc[0])
        captured_percent = 100.0 * captured / total if total else 0.0
        best_pred_part = eval_df[eval_df["pred_cluster"] == best_pred_cluster]
        extra_train_count = int((best_pred_part["gt_cluster"] != gt_cluster).sum())
        best_pred_cluster_size = int(len(best_pred_part))
        purity_percent = (
            100.0 * captured / best_pred_cluster_size
            if best_pred_cluster_size
            else 0.0
        )

        recovery_rows.append({
            "gt_cluster": gt_cluster,
            "best_pred_cluster": int(best_pred_cluster),
            "gt_train_count": total,
            "best_pred_cluster_size": best_pred_cluster_size,
            "captured_train_count": captured,
            "captured_percent": captured_percent,
            "missed_train_count": total - captured,
            "extra_train_count": extra_train_count,
            "purity_percent": purity_percent,
        })

    recovery = pd.DataFrame(recovery_rows)
    recovery = recovery.sort_values(
        ["captured_percent", "gt_train_count"],
        ascending=[False, False]
    )

    recovery_path = os.path.join(OUTPUT_DIR, "gt_cluster_recovery.txt")
    recovery.to_csv(recovery_path, index=False, sep="\t", encoding="utf-8-sig")

    print("GT cluster recovery kaydedildi:", recovery_path)
    print("\nGT cluster recovery preview:")
    for _, row in recovery.head(10).iterrows():
        print(
            f"{row['gt_cluster']} -> pred_cluster {int(row['best_pred_cluster'])} "
            f"captured {row['captured_percent']:.2f}% "
            f"({int(row['captured_train_count'])}/{int(row['gt_train_count'])}), "
            f"extra {int(row['extra_train_count'])}"
        )


def format_train_id_list(values):
    ids = []
    for x in values:
        if pd.isna(x):
            continue
        ids.append(int(x))

    ids = sorted(set(ids))
    return ",".join(str(x) for x in ids)


def write_train_group_listings(df):
    # Predicted clustering output. tarama number here is the predicted cluster id.
    pred_lines = []
    pred_summary = (
        df
        .dropna(subset=["pred_cluster"])
        .groupby("pred_cluster")
        .agg(
            emitter_majority=("emitter_id", mode_or_nan),
            train_ids=("pulse_train_id", format_train_id_list),
        )
        .reset_index()
        .sort_values(["emitter_majority", "pred_cluster"], na_position="last")
    )

    for _, row in pred_summary.iterrows():
        emitter = int(row["emitter_majority"]) if not pd.isna(row["emitter_majority"]) else "unknown"
        tarama = int(row["pred_cluster"])
        pred_lines.append(f"emitter{emitter}_tarama{tarama} are {row['train_ids']} trains")

    pred_path = os.path.join(OUTPUT_DIR, "predicted_train_groups.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pred_lines))
        if pred_lines:
            f.write("\n")

    print("Predicted train groups kaydedildi:", pred_path)

    # Ground-truth grouping for the available feature trains only.
    gt_df = df.dropna(subset=["emitter_id", "tarama_idx"]).copy()
    gt_lines = []

    if not gt_df.empty:
        gt_summary = (
            gt_df
            .groupby(["emitter_id", "tarama_idx"])
            .agg(train_ids=("pulse_train_id", format_train_id_list))
            .reset_index()
            .sort_values(["emitter_id", "tarama_idx"])
        )

        for _, row in gt_summary.iterrows():
            emitter = int(row["emitter_id"])
            tarama = int(row["tarama_idx"])
            gt_lines.append(f"emitter{emitter}_tarama{tarama} are {row['train_ids']} trains")

    gt_path = os.path.join(OUTPUT_DIR, "ground_truth_train_groups.txt")
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(gt_lines))
        if gt_lines:
            f.write("\n")

    print("Ground-truth train groups kaydedildi:", gt_path)


# ============================================================
# 9) MAIN
# ============================================================

def main():
    print("Darbe dizi feature dosyalarÄ± okunuyor...")
    merged_df = build_merged_dataset(FEATURE_DIR, SCAN_DIR)
    merged_path = os.path.join(OUTPUT_DIR, "merged_pulse_train_features.txt")
    merged_df.to_csv(merged_path, index=False, sep="\t", encoding="utf-8-sig")
    print("Available feature GT cluster sayisi:", merged_df["gt_cluster"].nunique())

    print("Merged dataset kaydedildi:", merged_path)
    print("Toplam satÄ±r:", len(merged_df))
    print("Amplitude eÅŸleÅŸmeyen satÄ±r:", merged_df["amplitude"].isna().sum())
    print("GT eÅŸleÅŸmeyen satÄ±r:", merged_df["gt_cluster"].isna().sum())

    print("\nOnline birth-death clustering baÅŸlÄ±yor...")
    clustered_df, clusters = online_birth_death_clustering(merged_df)

    clustered_path = os.path.join(OUTPUT_DIR, "clustered_pulse_train_features.txt")
    clustered_df.to_csv(clustered_path, index=False, sep="\t", encoding="utf-8-sig")

    print("Clustered dataset kaydedildi:", clustered_path)
    print("Predicted cluster count:", clustered_df["pred_cluster"].nunique())

    write_train_group_listings(clustered_df)

    evaluate_clustering(clustered_df)

    print("\nBitti.")


if __name__ == "__main__":
    main()

