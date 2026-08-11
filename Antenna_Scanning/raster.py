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
    raster_sector_scanning_centering_a_target,
    antenna_gain_calculator,
)


# ============================================================
# 2) CONFIG
# ============================================================

PATTERN_CSV_PATH = r"C:\Users\Bogazici\Desktop\parabolic_reflector_pattern.csv"

OUTPUT_DIR = r"C:\Users\Bogazici\Desktop\kodlar\generated_raster_1200_txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_FULL_SAMPLES = 800
NUM_CUT_SAMPLES = 400
NUM_SAMPLES = NUM_FULL_SAMPLES + NUM_CUT_SAMPLES

AREA_MIN_DEG = 20.0
AREA_MAX_DEG = 20.0

NUM_STEPS_MIN = 5
NUM_STEPS_MAX = 12

NOISE_STD_CHOICES = [0.004]

SEED = 42
np.random.seed(SEED)

# The old generator first creates each sample. Then one of these realistic
# observed-data modes is applied to that same sample. Total sample count stays 1200.
REALISTIC_MODE_PROBS = {
    "unchanged": 0.65,
    "tail_missing_realistic": 0.20,
    "middle_segment": 0.05,
    "split_segment": 0.10,
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

def wrap_elevation_0_360_to_physical(el_deg):
    """
    0-360 arası seçilen elevation'ı fiziksel -90/+90 aralığına çevirir.
    """
    el = float(el_deg) % 360.0

    if el <= 90.0:
        return el
    elif el <= 270.0:
        return 180.0 - el
    else:
        return el - 360.0


def save_sample_as_txt(filepath, sample, metadata):
    source_start_idx = int(metadata.get("source_start_idx", 0))

    with open(filepath, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"# {key} {value}\n")

        f.write("index gain\n")

        for idx, gain in enumerate(sample):
            f.write(f"{source_start_idx + idx} {gain:.8f}\n")


# ============================================================
# 5) TEK RASTER SAMPLE ÜRET
# ============================================================

def make_one_raster_sample(pattern, sample_idx, cut_mode):
    """
    cut_mode = False -> raster tamamlansın
    cut_mode = True  -> raster 2/3'ünde kesilsin
    """

    target_az_deg = float(np.random.uniform(0.0, 360.0))

    # Kullanıcının dediği gibi 0-360 seçiyoruz.
    target_el_raw_deg = float(np.random.uniform(0.0, 360.0))

    # Gain hesabı için fiziksel elevation'a çeviriyoruz.
    target_el_deg = wrap_elevation_0_360_to_physical(target_el_raw_deg)

    scanning_area_deg = float(np.random.uniform(AREA_MIN_DEG, AREA_MAX_DEG))

    num_steps_az = int(np.random.randint(NUM_STEPS_MIN, NUM_STEPS_MAX + 1))
    num_steps_el = int(np.random.randint(NUM_STEPS_MIN, NUM_STEPS_MAX + 1))

    full_length = num_steps_az * num_steps_el

    if cut_mode:
        max_positions = int(np.ceil((2.0 / 3.0) * full_length))
        max_positions = max(1, min(max_positions, full_length))
    else:
        max_positions = None

    scan_az_list, scan_el_list = raster_sector_scanning_centering_a_target(
        scanning_step_size_deg=1.0,
        scanning_area_deg=scanning_area_deg,
        target_azimuth_deg=target_az_deg,
        target_elevation_deg=target_el_deg,
        num_steps_az=num_steps_az,
        num_steps_el=num_steps_el,
        max_positions=max_positions,
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

    observed_length = len(gains)

    noise_std = float(np.random.choice(NOISE_STD_CHOICES))
    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=gains.shape,
    ).astype(np.float32)
    gains_noisy = gains + noise

    # Fiziksel olarak negatif gain istemiyorsak sadece alt sınırı 0 yapıyoruz.
    gains_noisy = np.maximum(gains_noisy, 0.0)

    metadata = {
        "sample_idx": sample_idx,
        "scan_type": "raster_scanning",
        "cut_mode": bool(cut_mode),
        "target_az_deg": target_az_deg,
        "target_el_raw_0_360_deg": target_el_raw_deg,
        "target_el_deg_physical": target_el_deg,
        "scanning_area_deg": scanning_area_deg,
        "num_steps_az": num_steps_az,
        "num_steps_el": num_steps_el,
        "full_length": full_length,
        "max_positions": max_positions if max_positions is not None else "None",
        "observed_length": observed_length,
        "noise_std": noise_std,
        "gain_min": float(np.min(gains_noisy)),
        "gain_max": float(np.max(gains_noisy)),
    }

    return gains_noisy, metadata


def slice_raster_sample(full_sample, base_metadata, sample_idx, partial_mode):
    full_length = len(full_sample)

    if partial_mode == "unchanged":
        start_idx = 0
        end_idx = full_length

    elif partial_mode == "tail_missing_realistic":
        # Real data often loses a moderate tail, e.g. about 30 missing out of 150.
        keep_ratio = float(np.random.uniform(0.75, 0.90))
        start_idx = 0
        end_idx = int(round(full_length * keep_ratio))

    elif partial_mode == "middle_segment":
        keep_ratio = float(np.random.uniform(0.45, 0.70))
        segment_len = max(1, int(round(full_length * keep_ratio)))
        start_idx = int(np.random.randint(0, max(1, full_length - segment_len + 1)))
        end_idx = start_idx + segment_len

    elif partial_mode == "split_segment":
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
        raise ValueError(f"Unknown partial_mode: {partial_mode}")

    partial_sample = full_sample[start_idx:end_idx]

    metadata = dict(base_metadata)
    base_cut_mode = metadata.get("cut_mode")
    metadata.update({
        "sample_idx": sample_idx,
        "base_cut_mode": base_cut_mode,
        "post_process_mode": partial_mode,
        "source_full_length": full_length,
        "source_start_idx": start_idx,
        "source_end_idx_exclusive": end_idx,
        "observed_length": len(partial_sample),
        "missing_before_count": start_idx,
        "missing_after_count": full_length - end_idx,
        "missing_total_count": full_length - len(partial_sample),
        "kept_ratio": len(partial_sample) / max(full_length, 1),
    })

    if partial_mode == "split_segment":
        metadata["split_part"] = split_part
    else:
        metadata["split_part"] = "None"

    if len(partial_sample) > 0:
        metadata["gain_min"] = float(np.min(partial_sample))
        metadata["gain_max"] = float(np.max(partial_sample))

    return partial_sample, metadata


def generate_realistic_partial_samples(pattern, start_sample_idx):
    partial_modes = []
    np.random.shuffle(partial_modes)

    metadata_rows = []

    for j, partial_mode in enumerate(partial_modes):
        sample_idx = start_sample_idx + j
        full_sample, base_meta = make_one_raster_sample(pattern, sample_idx, False)
        sample, meta = slice_raster_sample(full_sample, base_meta, sample_idx, partial_mode)

        filename = f"raster_{partial_mode}_{sample_idx:05d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        save_sample_as_txt(filepath, sample, meta)

        meta["filename"] = filename
        metadata_rows.append(meta)

        if (j + 1) % 100 == 0:
            print(f"{j + 1}/{len(partial_modes)} realistic partial raster txt sample Ã¼retildi.")

    return pd.DataFrame(metadata_rows)


def choose_post_process_mode():
    modes = list(REALISTIC_MODE_PROBS.keys())
    probs = np.array([REALISTIC_MODE_PROBS[m] for m in modes], dtype=float)
    probs = probs / probs.sum()
    return str(np.random.choice(modes, p=probs))


# ============================================================
# 6) MAIN
# ============================================================

def main():
    pattern = load_pattern_from_csv(PATTERN_CSV_PATH)

    metadata_rows = []

    sample_modes = [False] * NUM_FULL_SAMPLES + [True] * NUM_CUT_SAMPLES
    np.random.shuffle(sample_modes)

    for i, cut_mode in enumerate(sample_modes):
        sample, meta = make_one_raster_sample(pattern, i, cut_mode)
        post_process_mode = choose_post_process_mode()
        sample, meta = slice_raster_sample(sample, meta, i, post_process_mode)

        if cut_mode:
            filename = f"raster_cut_2over3_{i:05d}.txt"
        else:
            filename = f"raster_full_{i:05d}.txt"

        filepath = os.path.join(OUTPUT_DIR, filename)

        save_sample_as_txt(filepath, sample, meta)

        meta["filename"] = filename
        metadata_rows.append(meta)

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{NUM_SAMPLES} raster txt sample üretildi.")

    metadata_df = pd.DataFrame(metadata_rows)

    metadata_path = os.path.join(OUTPUT_DIR, "raster_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print("\nBitti kral.")
    print("Toplam sample:", NUM_SAMPLES)
    print("Full raster sample:", NUM_FULL_SAMPLES)
    print("Cut raster sample:", NUM_CUT_SAMPLES)
    print("Output folder:", OUTPUT_DIR)
    print("Metadata:", metadata_path)

    print("\nObserved length özeti:")
    print(metadata_df["observed_length"].describe())

    print("\nFull / cut sayısı:")
    print(metadata_df["cut_mode"].value_counts())

    print("\nGain max özeti:")
    print(metadata_df["gain_max"].describe())

    realistic_metadata_df = metadata_df
    realistic_metadata_path = metadata_path

    print("\nPost-processed raster sample:", len(realistic_metadata_df))
    print("Post-processed metadata:", realistic_metadata_path)

    print("\nRealistic partial mode sayÄ±sÄ±:")
    print(realistic_metadata_df["post_process_mode"].value_counts())

    print("\nRealistic observed length Ã¶zeti:")
    print(realistic_metadata_df["observed_length"].describe())


if __name__ == "__main__":
    main()
