import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TARGET_SCAN_TYPE = "raster_scanning"
DEFAULT_SCENARIO_ID = 9
DEFAULT_EMITTER_IDS = None
OUTPUT_ROOT = r"C:\Users\Bogazici\Desktop\kodlar\Antenna_Scanning"


# ============================================================
# HELPERS
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


def split_whitespace_line(line):
    return line.strip().split()


def split_table_line_2plus_spaces(line):
    return re.split(r"\s{2,}", line.strip())


def parse_emitter_id_from_filename(path):
    m = re.search(r"emitter_(\d+)\.txt$", os.path.basename(path), flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def parse_frame_time_and_freq(filename):
    base = os.path.basename(filename)

    m = re.search(r"db_(\d+)_([0-9.]+)\s*MHz", base, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), float(m.group(2))

    m_time = re.search(r"_(\d+)_", base)
    frame_time = int(m_time.group(1)) if m_time else np.nan

    m_freq = re.search(r"_([0-9.]+)\s*MHz", base, flags=re.IGNORECASE)
    file_freq_mhz = float(m_freq.group(1)) if m_freq else np.nan

    return frame_time, file_freq_mhz


def format_id_list(values):
    ids = []
    for x in values:
        if pd.isna(x):
            continue
        ids.append(int(x))
    return ",".join(str(x) for x in sorted(set(ids)))


def parse_emitter_ids(value):
    if value is None or str(value).strip() == "":
        return None

    emitter_ids = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            emitter_ids.append(int(part))

    return emitter_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare real raster antenna_scanning groups with captured Darbe Dizi feature trains."
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=DEFAULT_SCENARIO_ID,
        help="Scenario number under C:\\Users\\Bogazici\\Desktop\\senaryolar.",
    )
    parser.add_argument(
        "--emitters",
        default=None,
        help="Optional comma-separated emitter IDs, for example: 0,5,6",
    )
    return parser.parse_args()


def build_paths(scenario_id):
    scenario_dir = rf"C:\Users\Bogazici\Desktop\senaryolar\senaryo {scenario_id}"
    feature_dir = os.path.join(scenario_dir, "Darbe Dizi özellikleri")
    scan_dir = os.path.join(scenario_dir, "antenna_scanning")
    output_dir = os.path.join(
        OUTPUT_ROOT,
        f"raster_group_capture_analysis_outputs{scenario_id}",
    )
    return scenario_dir, feature_dir, scan_dir, output_dir


# ============================================================
# READ FEATURE FILES: AVAILABLE / CAPTURED PULSE TRAINS
# ============================================================

def read_single_feature_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        return None

    header = split_whitespace_line(lines[0].replace("#", "").strip())
    values = split_whitespace_line(lines[1].strip())

    if len(values) != len(header):
        print(f"[WARN] Skipping malformed feature file: {os.path.basename(file_path)}")
        print("       Header count:", len(header), "Value count:", len(values))
        return None

    data = dict(zip(header, values))
    frame_time, file_freq_mhz = parse_frame_time_and_freq(os.path.basename(file_path))

    return {
        "feature_filename": os.path.basename(file_path),
        "frame_time": frame_time,
        "file_freq_mhz": file_freq_mhz,
        "emitter_id": safe_int(data.get("emitter_id(GT)")),
        "pulse_train_id": safe_int(data.get("Darbe_Dizi_no")),
        "feature_antenna_scan_type_raw": safe_int(data.get("antenna_scan_type")),
        "operation_mode": data.get("operation_mode(GT)"),
        "modulation": safe_int(data.get("modulation")),
        "pri_mod_type": safe_int(data.get("pri_mod_type")),
        "toad": safe_float(data.get("TOAD")),
        "pw": safe_float(data.get("mean_pw")),
        "fc": safe_float(data.get("mean_fc")),
        "bw": safe_float(data.get("mean_bw")),
        "mean_snr_db": safe_float(data.get("mean_snr_db")),
    }


def read_all_feature_files(feature_dir):
    rows = []
    for file_path in sorted(glob.glob(os.path.join(feature_dir, "*.txt"))):
        row = read_single_feature_file(file_path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No feature files read from: {feature_dir}")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["emitter_id", "pulse_train_id"]).copy()
    df["emitter_id"] = df["emitter_id"].astype(int)
    df["pulse_train_id"] = df["pulse_train_id"].astype(int)
    return df


# ============================================================
# READ ANTENNA_SCANNING FILES: REAL SCAN GROUPS
# ============================================================

def read_single_scanning_file(file_path):
    emitter_id = parse_emitter_id_from_filename(file_path)
    if emitter_id is None:
        return pd.DataFrame()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]

    if len(lines) < 3:
        return pd.DataFrame()

    header = split_table_line_2plus_spaces(lines[0])
    data_lines = lines[2:]

    rows = []
    for line in data_lines:
        parts = split_table_line_2plus_spaces(line)
        if len(parts) != len(header):
            continue

        row = dict(zip(header, parts))
        row["emitter_id"] = emitter_id
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    col_map = {
        "Tarama Tipi": "scan_type",
        "Tarama idx": "scan_idx",
        "Darbe Dizi idx": "pulse_train_id",
        "Amplitude": "amplitude",
    }

    keep_cols = [col for col in col_map if col in df.columns] + ["emitter_id"]
    df = df[keep_cols].rename(columns=col_map)

    df["emitter_id"] = df["emitter_id"].apply(safe_int)
    df["scan_idx"] = df["scan_idx"].apply(safe_int)
    df["pulse_train_id"] = df["pulse_train_id"].apply(safe_int)

    if "amplitude" in df.columns:
        df["amplitude"] = df["amplitude"].apply(safe_float)
    else:
        df["amplitude"] = np.nan

    return df


def read_all_scanning_files(scan_dir, emitter_ids=None):
    rows = []
    emitter_filter = set(emitter_ids) if emitter_ids is not None else None

    for file_path in sorted(glob.glob(os.path.join(scan_dir, "emitter_*.txt"))):
        emitter_id = parse_emitter_id_from_filename(file_path)
        if emitter_filter is not None and emitter_id not in emitter_filter:
            continue

        df = read_single_scanning_file(file_path)
        if not df.empty:
            rows.append(df)

    if not rows:
        raise RuntimeError(f"No antenna_scanning files read from: {scan_dir}")

    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["emitter_id", "scan_idx", "pulse_train_id"]).copy()
    df["emitter_id"] = df["emitter_id"].astype(int)
    df["scan_idx"] = df["scan_idx"].astype(int)
    df["pulse_train_id"] = df["pulse_train_id"].astype(int)

    return df.drop_duplicates(
        subset=["emitter_id", "scan_idx", "pulse_train_id"],
        keep="first",
    ).reset_index(drop=True)


# ============================================================
# ANALYSIS
# ============================================================

def build_raster_comparison(feature_df, scan_df):
    raster_scan_df = scan_df[
        scan_df["scan_type"].astype(str).str.lower().eq(TARGET_SCAN_TYPE.lower())
    ].copy()

    feature_keys = set(zip(feature_df["emitter_id"], feature_df["pulse_train_id"]))

    rows = []
    for (emitter_id, scan_idx), part in raster_scan_df.groupby(["emitter_id", "scan_idx"]):
        real_ids = sorted(part["pulse_train_id"].astype(int).unique().tolist())
        captured_ids = [pid for pid in real_ids if (int(emitter_id), pid) in feature_keys]
        missing_ids = [pid for pid in real_ids if (int(emitter_id), pid) not in feature_keys]

        real_count = len(real_ids)
        captured_count = len(captured_ids)

        rows.append({
            "emitter_id": int(emitter_id),
            "scan_idx": int(scan_idx),
            "gt_cluster": f"emitter_{int(emitter_id)}_scan_{int(scan_idx)}",
            "scan_type": TARGET_SCAN_TYPE,
            "real_train_count": real_count,
            "captured_train_count": captured_count,
            "missing_train_count": len(missing_ids),
            "capture_percent": 100.0 * captured_count / real_count if real_count else 0.0,
            "real_train_ids": ",".join(str(x) for x in real_ids),
            "captured_train_ids": ",".join(str(x) for x in captured_ids),
            "missing_train_ids": ",".join(str(x) for x in missing_ids),
        })

    summary_columns = [
        "emitter_id",
        "scan_idx",
        "gt_cluster",
        "scan_type",
        "real_train_count",
        "captured_train_count",
        "missing_train_count",
        "capture_percent",
        "real_train_ids",
        "captured_train_ids",
        "missing_train_ids",
    ]
    summary = pd.DataFrame(rows, columns=summary_columns)
    if not summary.empty:
        summary = summary.sort_values(["emitter_id", "scan_idx"]).reset_index(drop=True)

    matched_columns = list(feature_df.columns) + ["scan_idx", "scan_type", "amplitude", "gt_cluster"]
    if raster_scan_df.empty:
        matched = pd.DataFrame(columns=matched_columns)
        return summary, matched, raster_scan_df

    matched = feature_df.merge(
        raster_scan_df[["emitter_id", "pulse_train_id", "scan_idx", "scan_type", "amplitude"]],
        on=["emitter_id", "pulse_train_id"],
        how="inner",
        validate="many_to_one",
    )
    matched["gt_cluster"] = (
        "emitter_" + matched["emitter_id"].astype(str)
        + "_scan_" + matched["scan_idx"].astype(str)
    )

    return summary, matched, raster_scan_df


def build_emitter_scan_type_overview(feature_df, scan_df):
    scan_overview = (
        scan_df
        .groupby(["emitter_id", "scan_type"])
        .agg(
            antenna_scanning_train_count=("pulse_train_id", "nunique"),
            scan_group_count=("scan_idx", "nunique"),
        )
        .reset_index()
    )

    feature_overview = (
        feature_df
        .groupby("emitter_id")
        .agg(feature_train_count=("pulse_train_id", "nunique"))
        .reset_index()
    )

    overview = scan_overview.merge(
        feature_overview,
        on="emitter_id",
        how="left",
    )
    overview["feature_train_count"] = overview["feature_train_count"].fillna(0).astype(int)

    return overview.sort_values(["emitter_id", "scan_type"]).reset_index(drop=True)


def write_captured_group_text(summary, output_dir):
    lines = []
    for _, row in summary.iterrows():
        captured_ids = row["captured_train_ids"]
        if captured_ids == "":
            captured_ids = "NONE"
        lines.append(
            f"emitter{int(row['emitter_id'])}_raster{int(row['scan_idx'])} "
            f"captured {int(row['captured_train_count'])}/{int(row['real_train_count'])}: "
            f"{captured_ids}"
        )

    path = os.path.join(output_dir, "raster_captured_train_groups.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    return path


def main():
    args = parse_args()
    emitter_ids = parse_emitter_ids(args.emitters)
    _, feature_dir, scan_dir, output_dir = build_paths(args.scenario)
    os.makedirs(output_dir, exist_ok=True)

    feature_df = read_all_feature_files(feature_dir)
    scan_df = read_all_scanning_files(scan_dir, emitter_ids=emitter_ids)

    summary, matched, raster_scan_df = build_raster_comparison(feature_df, scan_df)
    overview = build_emitter_scan_type_overview(feature_df, scan_df)

    summary_path = os.path.join(output_dir, "raster_group_capture_summary.csv")
    matched_path = os.path.join(output_dir, "raster_captured_feature_rows.csv")
    raw_raster_path = os.path.join(output_dir, "raster_real_antenna_scanning_rows.csv")
    overview_path = os.path.join(output_dir, "scenario_emitter_scan_type_overview.csv")

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    matched.to_csv(matched_path, index=False, encoding="utf-8-sig")
    raster_scan_df.to_csv(raw_raster_path, index=False, encoding="utf-8-sig")
    overview.to_csv(overview_path, index=False, encoding="utf-8-sig")
    text_path = write_captured_group_text(summary, output_dir)

    print("Raster group capture analysis complete.")
    print("Scenario:", args.scenario)
    print("Emitter filter:", emitter_ids if emitter_ids is not None else "all")
    print("Feature dir:", feature_dir)
    print("Antenna scanning dir:", scan_dir)
    print("Output dir:", output_dir)
    print()
    print("Raster GT groups:", len(summary))
    print("All antenna_scanning emitters:", sorted(scan_df["emitter_id"].unique().tolist()))
    print("Raster antenna_scanning emitters:", sorted(raster_scan_df["emitter_id"].unique().tolist()))
    print("Raster GT pulse trains:", len(raster_scan_df))
    print("Captured raster pulse trains:", len(matched))
    print("Missing raster pulse trains:", int(summary["missing_train_count"].sum()))
    print()
    print("Saved:")
    print(summary_path)
    print(matched_path)
    print(raw_raster_path)
    print(overview_path)
    print(text_path)


if __name__ == "__main__":
    main()
