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
    antenna_gain_calculator,
)


# ============================================================
# 2) CONFIG
# ============================================================

PATTERN_CSV_PATH = r"C:\Users\Bogazici\Desktop\parabolic_reflector_pattern.csv"

OUTPUT_DIR = r"C:\Users\Bogazici\Desktop\kodlar\generated_lock_on_1200_txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 1200

# Lock-on içinde pulse train sayısı: 20-90
MIN_PULSE_TRAIN_COUNT = 20
MAX_PULSE_TRAIN_COUNT = 90

# Target açıları
TARGET_AZ_MIN_DEG = 0.0
TARGET_AZ_MAX_DEG = 360.0

TARGET_EL_MIN_DEG = 10.0
TARGET_EL_MAX_DEG = 50.0

# Lock-on tam hedefte olmasın diye küçük sabit pointing error
# Bu hata scan boyunca SABİT kalır, yani yine lock-on olur.
POINTING_ERROR_AZ_MIN_DEG = -2.0
POINTING_ERROR_AZ_MAX_DEG = 2.0

POINTING_ERROR_EL_MIN_DEG = -2.0
POINTING_ERROR_EL_MAX_DEG = 2.0

SEED = 42
np.random.seed(SEED)


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

def wrap_azimuth_deg(az):
    return ((az % 360.0) + 360.0) % 360.0


def clip_elevation_deg(el):
    return float(np.clip(el, -90.0, 90.0))


def save_sample_as_txt(filepath, sample, metadata):
    with open(filepath, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"# {key} {value}\n")

        f.write("index gain\n")

        for idx, gain in enumerate(sample):
            f.write(f"{idx} {gain:.8f}\n")


# ============================================================
# 5) TEK LOCK-ON SAMPLE ÜRET
# ============================================================

def make_one_lock_on_sample(pattern, sample_idx):
    # Target random
    target_az_deg = float(np.random.uniform(TARGET_AZ_MIN_DEG, TARGET_AZ_MAX_DEG))
    target_el_deg = float(np.random.uniform(TARGET_EL_MIN_DEG, TARGET_EL_MAX_DEG))

    # Pulse train count random: 20-90
    pulse_train_count = int(
        np.random.randint(MIN_PULSE_TRAIN_COUNT, MAX_PULSE_TRAIN_COUNT + 1)
    )

    # Lock-on beam sabit durur.
    # Tam hedefe kilitli olsun ama ufak sabit pointing error ekleyelim.
    pointing_error_az_deg = float(
        np.random.uniform(POINTING_ERROR_AZ_MIN_DEG, POINTING_ERROR_AZ_MAX_DEG)
    )
    pointing_error_el_deg = float(
        np.random.uniform(POINTING_ERROR_EL_MIN_DEG, POINTING_ERROR_EL_MAX_DEG)
    )

    pointing_az_deg = wrap_azimuth_deg(target_az_deg + pointing_error_az_deg)
    pointing_el_deg = clip_elevation_deg(target_el_deg + pointing_error_el_deg)

    # Lock-on'da gain temiz halde sabit.
    clean_gain = antenna_gain_calculator(
        pattern,
        pointing_az_deg,
        pointing_el_deg,
        target_az_deg,
        target_el_deg,
    )

    gains = np.full(
        shape=pulse_train_count,
        fill_value=clean_gain,
        dtype=np.float32,
    )

    metadata = {
        "sample_idx": sample_idx,
        "scan_type": "lock_on",
        "target_az_deg": target_az_deg,
        "target_el_deg": target_el_deg,
        "pointing_az_deg": pointing_az_deg,
        "pointing_el_deg": pointing_el_deg,
        "pointing_error_az_deg": pointing_error_az_deg,
        "pointing_error_el_deg": pointing_error_el_deg,
        "pulse_train_count": pulse_train_count,
        "clean_gain": float(clean_gain),
        "gain_min": float(np.min(gains)),
        "gain_max": float(np.max(gains)),
        "gain_mean": float(np.mean(gains)),
    }

    return gains, metadata


# ============================================================
# 6) MAIN
# ============================================================

def main():
    pattern = load_pattern_from_csv(PATTERN_CSV_PATH)

    metadata_rows = []

    for i in range(NUM_SAMPLES):
        sample, meta = make_one_lock_on_sample(pattern, i)

        filename = f"lock_on_{i:05d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        save_sample_as_txt(filepath, sample, meta)

        meta["filename"] = filename
        metadata_rows.append(meta)

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{NUM_SAMPLES} lock-on txt sample üretildi.")

    metadata_df = pd.DataFrame(metadata_rows)

    metadata_path = os.path.join(OUTPUT_DIR, "lock_on_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print("\nBitti kral.")
    print("Toplam sample:", NUM_SAMPLES)
    print("Output folder:", OUTPUT_DIR)
    print("Metadata:", metadata_path)

    print("\nPulse train count özeti:")
    print(metadata_df["pulse_train_count"].describe())

    print("\nClean gain özeti:")
    print(metadata_df["clean_gain"].describe())


if __name__ == "__main__":
    main()
