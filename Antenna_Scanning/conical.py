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
    conical_beam_az_el_deg,
    antenna_gain_calculator,
)


# ============================================================
# 2) CONFIG
# ============================================================

PATTERN_CSV_PATH = r"C:\Users\Bogazici\Desktop\parabolic_reflector_pattern.csv"

OUTPUT_DIR = r"C:\Users\Bogazici\Desktop\kodlar\generated_conical_1200_txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 1200

# Conical toplam açı: 2-12 derece
MIN_CONE_TOTAL_ANGLE_DEG = 2.0
MAX_CONE_TOTAL_ANGLE_DEG = 5.0

# Her conical scanning içinde pulse train sayısı: 50-100
MIN_PULSE_TRAIN_COUNT = 50
MAX_PULSE_TRAIN_COUNT = 100

# Target açıları
TARGET_AZ_MIN_DEG = 0.0
TARGET_AZ_MAX_DEG = 360.0

TARGET_EL_MIN_DEG = 10.0
TARGET_EL_MAX_DEG = 50.0

NOISE_STD_CHOICES = [0.004]

SEED = 42
np.random.seed(SEED)

CONICAL_OBSERVED_MODE_PROBS = {
    "full_captured": 0.62,
    "split_segment": 0.20,
    "end_missing": 0.15,
    "middle_missing": 0.03,
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
    source_start_idx = int(metadata.get("source_start_idx", 0))

    with open(filepath, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"# {key} {value}\n")

        f.write("index gain\n")

        for idx, gain in enumerate(sample):
            f.write(f"{source_start_idx + idx} {gain:.8f}\n")


def choose_observed_mode():
    modes = list(CONICAL_OBSERVED_MODE_PROBS.keys())
    probs = np.array([CONICAL_OBSERVED_MODE_PROBS[m] for m in modes], dtype=float)
    probs = probs / probs.sum()
    return str(np.random.choice(modes, p=probs))


def apply_observed_mode(sample, metadata, observed_mode):
    full_length = len(sample)

    if observed_mode == "full_captured":
        start_idx = 0
        end_idx = full_length

    elif observed_mode == "end_missing":
        keep_ratio = float(np.random.uniform(0.75, 0.95))
        start_idx = 0
        end_idx = int(round(full_length * keep_ratio))

    elif observed_mode == "middle_missing":
        keep_ratio = float(np.random.uniform(0.55, 0.80))
        segment_len = max(1, int(round(full_length * keep_ratio)))
        start_idx = int(np.random.randint(0, max(1, full_length - segment_len + 1)))
        end_idx = start_idx + segment_len

    elif observed_mode == "split_segment":
        split_point = int(round(full_length * float(np.random.uniform(0.45, 0.60))))
        split_point = max(1, min(full_length - 1, split_point))

        if np.random.rand() < 0.5:
            start_idx = 0
            end_idx = split_point
            split_part = 0
        else:
            start_idx = split_point
            end_idx = full_length
            split_part = 1
    else:
        raise ValueError(f"Unknown observed_mode: {observed_mode}")

    observed_sample = sample[start_idx:end_idx]

    metadata = dict(metadata)
    metadata.update({
        "post_process_mode": observed_mode,
        "source_full_length": full_length,
        "source_start_idx": start_idx,
        "source_end_idx_exclusive": end_idx,
        "observed_length": len(observed_sample),
        "missing_before_count": start_idx,
        "missing_after_count": full_length - end_idx,
        "missing_total_count": full_length - len(observed_sample),
        "kept_ratio": len(observed_sample) / max(full_length, 1),
    })

    if observed_mode == "split_segment":
        metadata["split_part"] = split_part
    else:
        metadata["split_part"] = "None"

    if len(observed_sample) > 0:
        metadata["gain_min"] = float(np.min(observed_sample))
        metadata["gain_max"] = float(np.max(observed_sample))
        metadata["gain_mean"] = float(np.mean(observed_sample))

    return observed_sample, metadata


def make_conical_scan_exact_n_steps(
    num_steps,
    cone_total_angle_deg,
    target_az_deg,
    target_el_deg,
):
    """
    Tam olarak num_steps adet conical beam pozisyonu üretir.

    cone_total_angle_deg:
        Toplam conical açı.
        antenna_scanning_functions içindeki conical_beam_az_el_deg
        bunun yarısını gerçek cone half-angle olarak kullanır.

    num_steps:
        Bu scan içindeki pulse train sayısı.
    """

    scan_az_list = []
    scan_el_list = []

    step_deg = 360.0 / float(num_steps)

    for k in range(num_steps):
        az, el = conical_beam_az_el_deg(
            target_azimuth_deg=target_az_deg,
            target_elevation_deg=target_el_deg,
            scanning_area_deg=cone_total_angle_deg,
            step_index=k,
            scanning_step_size_deg=step_deg,
        )

        scan_az_list.append(az)
        scan_el_list.append(el)

    return scan_az_list, scan_el_list, step_deg


# ============================================================
# 5) TEK CONICAL SAMPLE ÜRET
# ============================================================

def make_one_conical_sample(pattern, sample_idx):
    # Target random
    target_az_deg = float(np.random.uniform(TARGET_AZ_MIN_DEG, TARGET_AZ_MAX_DEG))
    target_el_deg = float(np.random.uniform(TARGET_EL_MIN_DEG, TARGET_EL_MAX_DEG))

    # Toplam conical açı random: 2-12 derece
    cone_total_angle_deg = float(
        np.random.uniform(MIN_CONE_TOTAL_ANGLE_DEG, MAX_CONE_TOTAL_ANGLE_DEG)
    )

    # Pulse train count random: 50-100
    pulse_train_count = int(
        np.random.randint(MIN_PULSE_TRAIN_COUNT, MAX_PULSE_TRAIN_COUNT + 1)
    )

    scan_az_list, scan_el_list, step_deg = make_conical_scan_exact_n_steps(
        num_steps=pulse_train_count,
        cone_total_angle_deg=cone_total_angle_deg,
        target_az_deg=target_az_deg,
        target_el_deg=target_el_deg,
    )

    gains = []

    for scan_az_deg, scan_el_deg in zip(scan_az_list, scan_el_list):
        gain = antenna_gain_calculator(
            pattern,
            scan_az_deg,
            scan_el_deg,
            target_az_deg,
            target_el_deg,
        )
        gains.append(gain)

    gains = np.asarray(gains, dtype=np.float32)

    noise_std = float(np.random.choice(NOISE_STD_CHOICES))
    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=gains.shape,
    ).astype(np.float32)
    gains_noisy = gains + noise

    metadata = {
        "sample_idx": sample_idx,
        "scan_type": "conical_scanning",
        "target_az_deg": target_az_deg,
        "target_el_deg": target_el_deg,
        "cone_total_angle_deg": cone_total_angle_deg,
        "cone_half_angle_deg": cone_total_angle_deg / 2.0,
        "pulse_train_count": pulse_train_count,
        "step_deg": step_deg,
        "noise_std": noise_std,
        "gain_min": float(np.min(gains_noisy)),
        "gain_max": float(np.max(gains_noisy)),
        "gain_mean": float(np.mean(gains_noisy)),
    }

    return gains_noisy, metadata


# ============================================================
# 6) MAIN
# ============================================================

def main():
    pattern = load_pattern_from_csv(PATTERN_CSV_PATH)

    metadata_rows = []

    for i in range(NUM_SAMPLES):
        sample, meta = make_one_conical_sample(pattern, i)
        observed_mode = choose_observed_mode()
        sample, meta = apply_observed_mode(sample, meta, observed_mode)

        filename = f"conical_scanning_{i:05d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        save_sample_as_txt(filepath, sample, meta)

        meta["filename"] = filename
        metadata_rows.append(meta)

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{NUM_SAMPLES} conical txt sample üretildi.")

    metadata_df = pd.DataFrame(metadata_rows)

    metadata_path = os.path.join(OUTPUT_DIR, "conical_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print("\nBitti kral.")
    print("Toplam sample:", NUM_SAMPLES)
    print("Output folder:", OUTPUT_DIR)
    print("Metadata:", metadata_path)

    print("\nPulse train count özeti:")
    print(metadata_df["pulse_train_count"].describe())

    print("\nObserved length ozeti:")
    print(metadata_df["observed_length"].describe())

    print("\nObserved mode sayilari:")
    print(metadata_df["post_process_mode"].value_counts())

    print("\nConical toplam açı özeti:")
    print(metadata_df["cone_total_angle_deg"].describe())

    print("\nNoise std sayıları:")
    print(metadata_df["noise_std"].value_counts())

if __name__ == "__main__":
    main()
