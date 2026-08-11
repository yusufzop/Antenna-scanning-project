import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# 1) SENİN ANTENNA SCANNING KODLARININ OLDUĞU KLASÖR
# ============================================================

ANTENNA_CODE_DIR = r"C:\Users\Bogazici\Desktop\kodlar\Antenna_Scanning"

if ANTENNA_CODE_DIR not in sys.path:
    sys.path.insert(0, ANTENNA_CODE_DIR)


from antenna_scanning_functions import (
    circular_scanning,
    antenna_gain_calculator,
)


# ============================================================
# 2) CONFIG
# ============================================================

PATTERN_CSV_PATH = r"C:\Users\Bogazici\Desktop\parabolic_reflector_pattern.csv"

OUTPUT_DIR = r"C:\Users\Bogazici\Desktop\kodlar\generated_circular_1200_txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 1200

TARGET_EL_DEG = 0.0
BORESIGHT_EL_DEG = 0.0

MIN_STEP_DEG = 1.0
MAX_STEP_DEG = 10.0

NOISE_STD_CHOICES = [0.004]

SEED = 42
np.random.seed(SEED)

CIRCULAR_CAPTURE_SHAPE_PROBS = {
    "begin_end_captured": 0.93,
    "full_captured": 0.07,
}

CIRCULAR_BEGIN_END_OUTPUT_PROBS = {
    "combined_start_end": 0.30,
    "split_start_end": 0.70,
}


# ============================================================
# 3) PATTERN CSV OKUMA
# ============================================================

def load_pattern_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Pattern CSV bulunamadı:\n{csv_path}\n\n"
            "Dosyanın Desktop'ta olduğundan ve adının tam olarak "
            "parabolic_reflector_pattern.csv olduğundan emin ol."
        )

    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    required_cols = ["Az_deg", "El_deg", "Gain_dBi", "Gain_linear"]

    for col in required_cols:
        if col not in data.dtype.names:
            raise ValueError(
                f"CSV içinde '{col}' kolonu yok.\n"
                f"Bulunan kolonlar: {data.dtype.names}"
            )

    pattern = {
        "Az_deg": np.asarray(data["Az_deg"], dtype=float),
        "El_deg": np.asarray(data["El_deg"], dtype=float),
        "Gain_dBi": np.asarray(data["Gain_dBi"], dtype=float),
        "Gain_linear": np.asarray(data["Gain_linear"], dtype=float),
    }

    print("Pattern CSV okundu:")
    print(csv_path)
    print("Pattern nokta sayısı:", len(pattern["Az_deg"]))
    print("Gain min:", np.min(pattern["Gain_linear"]))
    print("Gain max:", np.max(pattern["Gain_linear"]))

    return pattern


# ============================================================
# 4) HELPERS
# ============================================================

def save_sample_as_txt(filepath, sample, metadata):
    source_positions_1based = str(metadata.get("captured_positions_1based", "")).strip()
    source_positions = []

    if source_positions_1based:
        source_positions = [
            int(pos) - 1
            for pos in source_positions_1based.split()
            if pos.strip().isdigit()
        ]

    if len(source_positions) != len(sample):
        source_positions = list(range(len(sample)))

    with open(filepath, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"# {key} {value}\n")

        f.write("index gain\n")

        for idx, gain in zip(source_positions, sample):
            f.write(f"{idx} {gain:.8f}\n")


# ============================================================
# 5) TEK CIRCULAR SAMPLE ÜRET
# ============================================================

def make_one_circular_sample(pattern, sample_idx):
    # Target azimuth ve boresight start azimuth eşit
    common_az_deg = float(np.random.uniform(0.0, 360.0))

    target_az_deg = common_az_deg
    boresight_start_az_deg = common_az_deg

    # Açı geçişi 1-10 derece arası random
    step_deg = float(np.random.uniform(MIN_STEP_DEG, MAX_STEP_DEG))

    # circular_scanning zaten 1 tam tur üretir ve durur
    scan_az_list, scan_el_list = circular_scanning(
        scanning_step_size_deg=step_deg,
        boresight_location_azimuth_deg_start=boresight_start_az_deg,
        boresight_location_elevation_deg_start=BORESIGHT_EL_DEG,
    )

    gains = []

    for scan_az_deg, scan_el_deg in zip(scan_az_list, scan_el_list):
        gain = antenna_gain_calculator(
            pattern,
            scan_az_deg,
            scan_el_deg,
            target_az_deg,
            TARGET_EL_DEG,
        )
        gains.append(gain)

    gains = np.asarray(gains, dtype=np.float32)

    pulse_train_count = len(gains)

    noise_std = float(np.random.choice(NOISE_STD_CHOICES))
    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=gains.shape,
    ).astype(np.float32)
    gains_noisy = gains + noise

    metadata = {
        "sample_idx": sample_idx,
        "scan_type": "circular_scanning",
        "common_az_deg": common_az_deg,
        "target_az_deg": target_az_deg,
        "target_el_deg": TARGET_EL_DEG,
        "boresight_start_az_deg": boresight_start_az_deg,
        "boresight_el_deg": BORESIGHT_EL_DEG,
        "step_deg": step_deg,
        "noise_std": noise_std,
        "gain_min": float(np.min(gains_noisy)),
        "gain_max": float(np.max(gains_noisy)),
        "pulse_train_count": pulse_train_count,
    }

    return gains_noisy, metadata


def choose_from_probs(prob_map):
    modes = list(prob_map.keys())
    probs = np.asarray([prob_map[mode] for mode in modes], dtype=float)
    probs = probs / probs.sum()
    return str(np.random.choice(modes, p=probs))


def _blocks_from_positions(positions):
    if not positions:
        return []

    blocks = []
    start = positions[0]
    prev = positions[0]

    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        blocks.append((start, prev))
        start = pos
        prev = pos

    blocks.append((start, prev))
    return blocks


def _format_blocks_1based(blocks):
    return ";".join(f"{start + 1}-{end + 1}" for start, end in blocks)


def build_begin_end_positions(full_length):
    total_keep = int(round(full_length * np.random.uniform(0.22, 0.38)))
    total_keep = max(4, min(full_length, total_keep))

    start_len = max(2, int(round(total_keep * np.random.uniform(0.42, 0.58))))
    end_len = max(2, total_keep - start_len)

    if start_len + end_len >= full_length:
        start_positions = list(range(full_length))
        end_positions = []
    else:
        start_positions = list(range(start_len))
        end_positions = list(range(full_length - end_len, full_length))

    return start_positions, end_positions


def make_observed_outputs(sample, metadata):
    full_length = len(sample)
    capture_shape = choose_from_probs(CIRCULAR_CAPTURE_SHAPE_PROBS)

    if full_length <= 3 or capture_shape == "full_captured":
        return [
            make_output_from_positions(
                sample,
                metadata,
                list(range(full_length)),
                capture_shape,
                "full",
            )
        ]

    start_positions, end_positions = build_begin_end_positions(full_length)
    combined_positions = start_positions + end_positions
    output_mode = choose_from_probs(CIRCULAR_BEGIN_END_OUTPUT_PROBS)

    if output_mode == "combined_start_end" or not end_positions:
        return [
            make_output_from_positions(
                sample,
                metadata,
                combined_positions,
                capture_shape,
                "combined_start_end",
            )
        ]

    return [
        make_output_from_positions(
            sample,
            metadata,
            start_positions,
            capture_shape,
            "split_start",
        ),
        make_output_from_positions(
            sample,
            metadata,
            end_positions,
            capture_shape,
            "split_end",
        ),
    ]


def make_output_from_positions(sample, metadata, kept_positions, capture_shape, output_mode):
    full_length = len(sample)
    kept_positions = sorted(set(pos for pos in kept_positions if 0 <= pos < full_length))
    observed_sample = sample[kept_positions]
    blocks = _blocks_from_positions(kept_positions)
    missing_count = full_length - len(kept_positions)

    output_metadata = dict(metadata)
    output_metadata.update({
        "source_pulse_train_count": metadata.get("pulse_train_count", full_length),
        "pulse_train_count": len(observed_sample),
        "capture_shape": capture_shape,
        "post_process_mode": output_mode,
        "source_full_length": full_length,
        "observed_length": len(observed_sample),
        "captured_block_count": len(blocks),
        "captured_blocks_1based": _format_blocks_1based(blocks),
        "captured_positions_1based": " ".join(str(pos + 1) for pos in kept_positions),
        "missing_total_count": missing_count,
        "kept_ratio": float(len(observed_sample) / full_length) if full_length else 0.0,
    })

    if len(observed_sample) > 0:
        output_metadata["gain_min"] = float(np.min(observed_sample))
        output_metadata["gain_max"] = float(np.max(observed_sample))

    return observed_sample, output_metadata


# ============================================================
# 6) MAIN
# ============================================================

def main():
    pattern = load_pattern_from_csv(PATTERN_CSV_PATH)

    metadata_rows = []

    source_idx = 0

    while len(metadata_rows) < NUM_SAMPLES:
        sample, meta = make_one_circular_sample(pattern, source_idx)
        observed_outputs = make_observed_outputs(sample, meta)

        for observed_sample, observed_meta in observed_outputs:
            if len(metadata_rows) >= NUM_SAMPLES:
                break

            output_idx = len(metadata_rows)
            filename = f"circular_scanning_{output_idx:05d}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)

            observed_meta["sample_idx"] = output_idx
            observed_meta["source_sample_idx"] = source_idx

            save_sample_as_txt(filepath, observed_sample, observed_meta)

            observed_meta["filename"] = filename
            metadata_rows.append(observed_meta)

            if len(metadata_rows) % 100 == 0:
                print(f"{len(metadata_rows)}/{NUM_SAMPLES} circular txt sample uretiliyor.")

        source_idx += 1

    metadata_df = pd.DataFrame(metadata_rows)

    metadata_path = os.path.join(OUTPUT_DIR, "circular_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print("\nBitti kral.")
    print("Toplam sample:", NUM_SAMPLES)
    print("Output folder:", OUTPUT_DIR)
    print("Metadata:", metadata_path)

    print("\nPulse train count özeti:")
    print(metadata_df["pulse_train_count"].describe())
    print("\nObserved circular capture modes:")
    print(metadata_df["post_process_mode"].value_counts())
    print("\nCircular capture shapes:")
    print(metadata_df["capture_shape"].value_counts())
    print("\nObserved length ozeti:")
    print(metadata_df["observed_length"].describe())


if __name__ == "__main__":
    main()
