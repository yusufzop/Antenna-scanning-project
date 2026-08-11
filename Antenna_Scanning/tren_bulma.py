import os
import re
import pandas as pd


SCAN_DIR = r"C:\Users\Bogazici\Desktop\COG_RADAR_DATASET_senaryo_7\COG_RADAR_DATASET_senaryo_7\module_0\antenna_scanning"


def split_table_line(line):
    """
    Kolon isimlerinde boşluk olduğu için 2 veya daha fazla boşluğa göre split ediyoruz.
    Örn:
    Tarama Tipi        Tarama idx  Adım (°)  Darbe Dizi idx ...
    """
    return re.split(r"\s{2,}", line.strip())


def read_emitter_file(file_path, emitter_id):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f.readlines() if line.strip()]

    if len(lines) < 3:
        return pd.DataFrame()

    header = split_table_line(lines[0])
    data_lines = lines[2:]  # 0: header, 1: dashed separator, sonrası data

    rows = []

    for line in data_lines:
        parts = split_table_line(line)

        if len(parts) != len(header):
            continue

        row = dict(zip(header, parts))
        row["emitter_id"] = emitter_id
        row["emitter_file"] = f"emitter_{emitter_id}.txt"
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Sayısal kolonları dönüştür
    numeric_cols = [
        "Tarama idx",
        "Adım (°)",
        "Darbe Dizi idx",
        "Anten Kazancı",
        "Amplitude",
        "Emitter Poz Az (°)",
        "Emitter Poz El (°)",
        "Emitter Boresight Az (°)",
        "Emitter Boresight El (°)",
        "Açı Farkı Az (°)",
        "Açı Farkı El (°)",
        "Off-Axis (°)",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all_scanning_files(scan_dir):
    all_dfs = []

    for emitter_id in range(5):
        file_path = os.path.join(scan_dir, f"emitter_{emitter_id}.txt")

        if not os.path.exists(file_path):
            print(f"Bulunamadı: {file_path}")
            continue

        df = read_emitter_file(file_path, emitter_id)

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("Hiçbir emitter dosyası okunamadı.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    return full_df


def find_train_info(train_no, scan_df):
    train_no = int(train_no)

    matches = scan_df[scan_df["Darbe Dizi idx"] == train_no].copy()

    if matches.empty:
        print(f"\nDarbe Dizi idx = {train_no} bulunamadı.")
        return

    if len(matches) > 1:
        print(f"\nUyarı: Darbe Dizi idx = {train_no} için birden fazla eşleşme bulundu.")
        print("Sen unique demiştin; yine de hepsini gösteriyorum.\n")

    for _, row in matches.iterrows():
        print("\n==============================")
        print(f"Darbe Dizi idx : {int(row['Darbe Dizi idx'])}")
        print(f"Emitter        : emitter_{int(row['emitter_id'])}")
        print(f"Tarama Tipi    : {row['Tarama Tipi']}")
        print(f"Tarama idx     : {int(row['Tarama idx'])}")

        if "Amplitude" in row and pd.notna(row["Amplitude"]):
            print(f"Amplitude      : {row['Amplitude']}")

        print("==============================")


def main():
    print("Antenna scanning dosyaları okunuyor...")
    scan_df = load_all_scanning_files(SCAN_DIR)

    print(f"Toplam okunan satır: {len(scan_df)}")
    print("Hazır. Darbe dizi numarası gir.")

    while True:
        user_input = input("\nDarbe Dizi idx gir, çıkmak için q: ").strip()

        if user_input.lower() in ["q", "quit", "exit"]:
            break

        if not user_input.isdigit():
            print("Sayı girmen lazım.")
            continue

        find_train_info(user_input, scan_df)


if __name__ == "__main__":
    main()