import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


BASE_SCENARIO_DIR = r"C:\Users\Bogazici\Desktop\senaryolar"
OUTPUT_ROOT = r"C:\Users\Bogazici\Desktop\kodlar\Antenna_Scanning"
TARGET_SCAN_TYPE = "circular_scanning"


def split_table_line_2plus_spaces(line):
    return re.split(r"\s{2,}", line.strip())


def safe_int(x):
    try:
        if pd.isna(x):
            return np.nan
        return int(float(x))
    except Exception:
        return np.nan


def normalize_label(x):
    if pd.isna(x):
        return ""
    return re.sub(r"[\s\-]+", "_", str(x).strip().lower())


def parse_args():
    parser = argparse.ArgumentParser(
        description="List captured sequence positions inside real circular scanning groups."
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help="Optional scenario number. If omitted, scenarios 1..9 are processed.",
    )
    return parser.parse_args()


def read_single_scanning_file(file_path, scenario_id, emitter_id):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]

    if len(lines) < 3:
        return pd.DataFrame()

    header = split_table_line_2plus_spaces(lines[0])
    rows = []

    for line in lines[2:]:
        parts = split_table_line_2plus_spaces(line)
        if len(parts) != len(header):
            continue

        row = dict(zip(header, parts))
        rows.append({
            "scenario_id": scenario_id,
            "emitter_id": emitter_id,
            "scan_type": row.get("Tarama Tipi"),
            "scan_idx": safe_int(row.get("Tarama idx")),
            "pulse_train_id": safe_int(row.get("Darbe Dizi idx")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.dropna(subset=["scan_idx", "pulse_train_id"]).copy()


def read_scanning_files(scenario_id):
    scan_dir = os.path.join(BASE_SCENARIO_DIR, f"senaryo {scenario_id}", "antenna_scanning")
    rows = []

    for file_path in sorted(glob.glob(os.path.join(scan_dir, "emitter_*.txt"))):
        m = re.search(r"emitter_(\d+)\.txt$", os.path.basename(file_path), flags=re.IGNORECASE)
        if not m:
            continue

        emitter_id = int(m.group(1))
        df = read_single_scanning_file(file_path, scenario_id, emitter_id)
        if not df.empty:
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["emitter_id"] = df["emitter_id"].astype(int)
    df["scan_idx"] = df["scan_idx"].astype(int)
    df["pulse_train_id"] = df["pulse_train_id"].astype(int)

    return df


def read_single_feature_file(file_path, scenario_id):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        return None

    header = lines[0].replace("#", "").strip().split()
    values = lines[1].strip().split()

    if len(header) != len(values):
        return None

    data = dict(zip(header, values))
    return {
        "scenario_id": scenario_id,
        "emitter_id": safe_int(data.get("emitter_id(GT)")),
        "pulse_train_id": safe_int(data.get("Darbe_Dizi_no")),
    }


def read_feature_files(scenario_id):
    feature_dir = os.path.join(BASE_SCENARIO_DIR, f"senaryo {scenario_id}", "Darbe Dizi özellikleri")
    rows = []

    for file_path in sorted(glob.glob(os.path.join(feature_dir, "*.txt"))):
        row = read_single_feature_file(file_path, scenario_id)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["emitter_id", "pulse_train_id"]).copy()
    df["emitter_id"] = df["emitter_id"].astype(int)
    df["pulse_train_id"] = df["pulse_train_id"].astype(int)

    return df


def format_positions(values):
    return ",".join(str(int(x)) for x in values)


def analyze_scenario(scenario_id):
    scan_df = read_scanning_files(scenario_id)
    feature_df = read_feature_files(scenario_id)

    if scan_df.empty:
        return pd.DataFrame()

    circular_df = scan_df[
        scan_df["scan_type"].apply(normalize_label).eq(TARGET_SCAN_TYPE)
    ].copy()

    if circular_df.empty:
        return pd.DataFrame()

    feature_keys = set()
    if not feature_df.empty:
        feature_keys = set(zip(feature_df["emitter_id"], feature_df["pulse_train_id"]))

    rows = []
    for (emitter_id, scan_idx), part in circular_df.groupby(["emitter_id", "scan_idx"]):
        part = part.sort_values("pulse_train_id").reset_index(drop=True)
        part["sequence_index_1based"] = np.arange(1, len(part) + 1)
        part["is_captured"] = part.apply(
            lambda row: (int(row["emitter_id"]), int(row["pulse_train_id"])) in feature_keys,
            axis=1,
        )

        captured = part[part["is_captured"]]
        missing = part[~part["is_captured"]]

        rows.append({
            "scenario_id": scenario_id,
            "emitter_id": int(emitter_id),
            "scan_idx": int(scan_idx),
            "gt_cluster": f"emitter_{int(emitter_id)}_circular_{int(scan_idx)}",
            "real_length": len(part),
            "captured_count": len(captured),
            "missing_count": len(missing),
            "capture_percent": 100.0 * len(captured) / len(part) if len(part) else 0.0,
            "captured_positions_1based": format_positions(captured["sequence_index_1based"].tolist()),
            "missing_positions_1based": format_positions(missing["sequence_index_1based"].tolist()),
            "captured_train_ids": format_positions(captured["pulse_train_id"].tolist()),
            "missing_train_ids": format_positions(missing["pulse_train_id"].tolist()),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    scenario_ids = [args.scenario] if args.scenario is not None else list(range(1, 10))

    outputs = []
    for scenario_id in scenario_ids:
        df = analyze_scenario(scenario_id)
        if df.empty:
            continue
        outputs.append(df)

    if outputs:
        result = pd.concat(outputs, ignore_index=True)
    else:
        result = pd.DataFrame()

    suffix = str(args.scenario) if args.scenario is not None else "all"
    output_dir = os.path.join(OUTPUT_ROOT, f"circular_capture_position_outputs{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "circular_capture_positions.csv")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    text_path = os.path.join(output_dir, "circular_captured_positions.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for _, row in result.iterrows():
            captured = row["captured_positions_1based"]
            if pd.isna(captured) or captured == "":
                captured = "NONE"
            f.write(
                f"scenario{int(row['scenario_id'])}_emitter{int(row['emitter_id'])}"
                f"_circular{int(row['scan_idx'])} "
                f"captured {int(row['captured_count'])}/{int(row['real_length'])} "
                f"positions: {captured}\n"
            )

    print("Circular capture position analysis complete.")
    print("Scenarios:", scenario_ids)
    print("Circular groups:", len(result))
    if not result.empty:
        print("Total circular trains:", int(result["real_length"].sum()))
        print("Captured circular trains:", int(result["captured_count"].sum()))
        print("Missing circular trains:", int(result["missing_count"].sum()))
    print("Saved:", output_path)
    print("Saved:", text_path)


if __name__ == "__main__":
    main()
