"""
generate_scan_patterns.py

Generate the antenna-gain-vs-scan-step "signature" for each antenna scanning
type used in this project, sampling the scan at a 1-degree step, and save each
pattern to its own .npy file.

The value stored per scan step is the antenna gain toward a fixed target (the
receiver), in LINEAR scale, normalized so the on-boresight peak == 1.0. This is
the amplitude-like envelope an ESM receiver observes as the beam sweeps past it.

Scan types produced (filename -> project mapping):
    circular_360   full 360 deg azimuth revolution     -> circular_scanning
    circular_90    90 deg azimuth sector (+/-45 deg)    -> unidirectional_sector_scanning(area=90)
    raster_20x20   20x20 deg serpentine raster          -> raster_sector_scanning_centering_a_target(area=20)
    conical_3      conical scan, full cone width 3 deg  -> conical_scanning_centering_a_target(area=3)
    conical_5      conical scan, full cone width 5 deg  -> ...(area=5)
    conical_10     conical scan, full cone width 10 deg -> ...(area=10)
    lock_on        beam fixed on target (constant peak) -> lock_on

Notes on the "circular" interpretation:
    The live sim's circular_scanning() ignores scanning_area_deg and always
    sweeps a full 360 deg, so "circular 90" and "circular 360" would otherwise be
    identical. Here, by request, circular_90 is rendered as a real 90 deg azimuth
    sector so the two produce distinct signatures.

All scans are centered on the target direction (TARGET_AZ_DEG, TARGET_EL_DEG),
matching how emitter_class.set_scanning_type() points the beam at the receiver.

Usage:
    python generate_scan_patterns.py [output_dir]
    output_dir defaults to E:\\scan_patterns
"""
import os
import sys
import numpy as np

# --- Make the project's antenna_scanning_functions importable -----------------
PROJECT_DIR = r"c:\Users\Onur\Desktop\BRS\Bilissel_Frame_Generation"
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, PROJECT_DIR):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from antenna_scanning_functions import (
    load_radiation_pattern,
    circular_scanning,
    unidirectional_sector_scanning,
    raster_sector_scanning_centering_a_target,
    conical_scanning_centering_a_target,
    antenna_gain_calculator,
)

# --- Configuration ------------------------------------------------------------
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r"E:\scan_patterns"
RADIATION_PATTERN = "parabolic_reflector"  # the pattern every scenario emitter uses
STEP_DEG = 1.0                             # 1 degree per scan step (as requested)
TARGET_AZ_DEG = 0.0                        # receiver direction; scans are centered here
TARGET_EL_DEG = 0.0
LOCK_ON_LENGTH = 360                       # samples for the (constant) lock-on pattern
MAKE_OVERVIEW_PLOT = True                  # also save an overview PNG beside the .npy files


# --- Gain pattern -------------------------------------------------------------
def gain_pattern_normalized(pattern, scan_az_list, scan_el_list, pattern_max):
    """Antenna gain toward (TARGET_AZ_DEG, TARGET_EL_DEG) at each beam position,
    in linear scale, normalized so the pattern's boresight peak == 1.0.

    Uses the project's own antenna_gain_calculator (2-nearest-neighbour
    interpolation of the radiation pattern), the same routine emitter_class
    uses per frame, so these envelopes match the simulator."""
    gains = np.empty(len(scan_az_list), dtype=float)
    for i, (az, el) in enumerate(zip(scan_az_list, scan_el_list)):
        g = antenna_gain_calculator(pattern, az, el, TARGET_AZ_DEG, TARGET_EL_DEG)
        gains[i] = (g / pattern_max) if pattern_max > 0 else 0.0
    return gains


# --- Beam-position lists for each scan type (1 deg step) ----------------------
def build_scans():
    scans = {}

    # circular_360: full 360 deg azimuth revolution at fixed elevation.
    # Start 180 deg before the target az so the main lobe lands mid-array.
    az, el = circular_scanning(
        scanning_step_size_deg=STEP_DEG,
        boresight_location_azimuth_deg_start=TARGET_AZ_DEG - 180.0,
        boresight_location_elevation_deg_start=TARGET_EL_DEG,
    )
    scans["circular_360"] = (az, el)

    # circular_90: 90 deg azimuth sector centered on the target (+/-45 deg).
    az, el = unidirectional_sector_scanning(
        scanning_step_size_deg=STEP_DEG,
        scanning_area_deg=90.0,
        Target_azimuth_deg=TARGET_AZ_DEG,
        Target_elevation_deg=TARGET_EL_DEG,
    )
    scans["circular_90"] = (az, el)

    # raster_20x20: 20x20 deg square serpentine raster centered on the target.
    az, el = raster_sector_scanning_centering_a_target(
        scanning_step_size_deg=STEP_DEG,
        scanning_area_deg=20.0,
        target_azimuth_deg=TARGET_AZ_DEG,
        target_elevation_deg=TARGET_EL_DEG,
    )
    scans["raster_20x20"] = (az, el)

    # conical_{3,5,10}: beam rotates on a small circle (diameter = full cone
    # width = area) around the target; 1 deg phase step -> 360 positions/rev.
    for cone in (3.0, 5.0, 10.0):
        az, el = conical_scanning_centering_a_target(
            scanning_step_size_deg=STEP_DEG,
            scanning_area_deg=cone,
            target_azimuth_deg=TARGET_AZ_DEG,
            target_elevation_deg=TARGET_EL_DEG,
        )
        scans[f"conical_{int(cone)}"] = (az, el)

    # lock_on: beam fixed exactly on the target -> constant boresight gain.
    az = [TARGET_AZ_DEG] * LOCK_ON_LENGTH
    el = [TARGET_EL_DEG] * LOCK_ON_LENGTH
    scans["lock_on"] = (az, el)

    return scans


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pattern = load_radiation_pattern(RADIATION_PATTERN)
    pattern_max = float(np.max(pattern["Gain_linear"]))

    scans = build_scans()

    print(f"Radiation pattern : {RADIATION_PATTERN} (peak linear gain = {pattern_max:.4g})")
    print(f"Output folder     : {OUTPUT_DIR}")
    print(f"Scan step         : {STEP_DEG} deg")
    print(f"Target (receiver) : az={TARGET_AZ_DEG} deg, el={TARGET_EL_DEG} deg")
    print("-" * 72)
    print(f"{'file':<18}{'n_steps':>9}{'min':>11}{'max':>11}{'mean':>11}")
    print("-" * 72)

    results = {}
    for name, (az, el) in scans.items():
        g = gain_pattern_normalized(pattern, az, el, pattern_max)
        out_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
        np.save(out_path, g)
        results[name] = g
        print(f"{name:<18}{len(g):>9}{g.min():>11.5f}{g.max():>11.5f}{g.mean():>11.5f}")

    print("-" * 72)
    print(f"Saved {len(results)} pattern files (.npy) to {OUTPUT_DIR}")

    if MAKE_OVERVIEW_PLOT:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = list(results.keys())
            fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.6 * len(names)))
            axes = np.atleast_1d(axes)
            for ax, name in zip(axes, names):
                ax.plot(results[name], lw=1.2)
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
                ax.grid(True, ls=":", alpha=0.5)
                ax.set_ylim(-0.05, 1.05)
            axes[-1].set_xlabel("scan step (1 deg each)")
            fig.suptitle("Antenna gain toward target vs scan step (normalized, peak = 1)")
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            png = os.path.join(OUTPUT_DIR, "scan_patterns_overview.png")
            fig.savefig(png, dpi=120)
            plt.close(fig)
            print(f"Overview plot saved to {png}")
        except Exception as exc:  # plotting is a convenience, never fatal
            print(f"(overview plot skipped: {exc})")


if __name__ == "__main__":
    main()
