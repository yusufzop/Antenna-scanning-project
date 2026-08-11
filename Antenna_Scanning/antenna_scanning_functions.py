import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)
from scipy.interpolate import RegularGridInterpolator

def load_radiation_pattern(antenna_name: str):
    name = antenna_name.strip().lower()
    filename = f"{name}_pattern.csv"
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "antenna_radiation_patterns", filename)

    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    return {
        "Az_deg": np.asarray(data["Az_deg"], dtype=float),
        "El_deg": np.asarray(data["El_deg"], dtype=float),
        "Gain_dBi": np.asarray(data["Gain_dBi"], dtype=float),
        "Gain_linear": np.asarray(data["Gain_linear"], dtype=float),
    }

def  circular_scanning(scanning_step_size_deg,
boresight_location_azimuth_deg_start=0,
boresight_location_elevation_deg_start=0):
    # Başlangıç azimutundan itibaren TAM bir tur (360°) atar; mutlak 0–360 sınırıyla
    # kesilmez, böylece huzme/adım sayısı başlangıç açısından BAĞIMSIZDIR.
    boresight_location_elevation_deg=boresight_location_elevation_deg_start
    scanning_list_azimuth_deg=[]
    scanning_list_elevation_deg=[]
    n_steps = max(1, int(round(360.0 / scanning_step_size_deg)))
    for k in range(n_steps):
        az = boresight_location_azimuth_deg_start + k * scanning_step_size_deg
        scanning_list_azimuth_deg.append(((az % 360.0) + 360.0) % 360.0)
        scanning_list_elevation_deg.append(boresight_location_elevation_deg)
    return scanning_list_azimuth_deg, scanning_list_elevation_deg

def  unidirectional_sector_scanning(scanning_step_size_deg,
    scanning_area_deg,
    Target_azimuth_deg=0, 
    Target_elevation_deg=0
    ):
    """
    Tek yönlü sektör taraması.
    scanning_area_deg: sektör genişliği (°). None verilirse 360° kabul edilir.
    """
    if scanning_area_deg is None:
        scanning_area_deg = 360.0
    boresight_location_azimuth_deg=Target_azimuth_deg
    boresight_location_elevation_deg=Target_elevation_deg
    scanning_list_azimuth_deg=[]
    scanning_list_elevation_deg=[]
    start_az=boresight_location_azimuth_deg
    half_sector=scanning_area_deg/2
    sector_start=-half_sector
    steps=int(np.floor(scanning_area_deg/scanning_step_size_deg))+1
    for k in range(steps):
        az=sector_start+k*scanning_step_size_deg
        az=((az%360)+360)%360
        scanning_list_azimuth_deg.append(az)
        scanning_list_elevation_deg.append(boresight_location_elevation_deg)
    return scanning_list_azimuth_deg, scanning_list_elevation_deg

def  bidirectional_sector_scanning(scanning_step_size_deg,
    scanning_area_deg,
    boresight_location_azimuth_deg_start=0,
    boresight_location_elevation_deg_start=0):
    """
    İki yönlü (ping-pong) sektör taraması.
    scanning_area_deg: sektör genişliği (°). None verilirse 360° kabul edilir.
    """
    if scanning_area_deg is None:
        scanning_area_deg = 360.0
    boresight_location_azimuth_deg=boresight_location_azimuth_deg_start
    boresight_location_elevation_deg=boresight_location_elevation_deg_start
    scanning_list_azimuth_deg=[]
    scanning_list_elevation_deg=[]
    start_az=boresight_location_azimuth_deg
    half_sector=scanning_area_deg/2
    sector_start=start_az-half_sector
    sector_end=start_az+half_sector
    steps=int(np.floor(scanning_area_deg/scanning_step_size_deg))+1

    forward=[]
    for k in range(steps):
        az=sector_start+k*scanning_step_size_deg
        az=((az%360)+360)%360
        forward.append(az)

    # Backward sweep (end -> start), excluding endpoints to avoid duplicates
    backward=[]
    for k in range(steps-2, 0, -1):
        az=sector_start+k*scanning_step_size_deg
        az=((az%360)+360)%360
        backward.append(az)

    ping_pong = forward + backward
    for az in ping_pong:
        scanning_list_azimuth_deg.append(az)
        scanning_list_elevation_deg.append(boresight_location_elevation_deg)
    return scanning_list_azimuth_deg, scanning_list_elevation_deg

def  raster_scanning(scanning_step_size_deg,
    scanning_area_deg,
    boresight_location_azimuth_deg_start=0,
    boresight_location_elevation_deg_start=0):
    """
    Kare raster tarama: azimuth \\in [0, scanning_area_deg], elevation \\in [0, scanning_area_deg].
    scanning_area_deg: 1D kenar uzunluğu (°); toplam alan = scanning_area_deg x scanning_area_deg.
    boresight_* argümanları bilinçli olarak kullanılmaz (sabit ileri/yukarı arama sektörü).
    """
    if scanning_step_size_deg <= 0:
        raise ValueError("scanning_step_size_deg must be positive")
    if scanning_area_deg is None:
        scanning_area_deg = 90.0
    if scanning_area_deg <= 0:
        raise ValueError("scanning_area_deg must be positive")

    scanning_list_azimuth_deg=[]
    scanning_list_elevation_deg=[]

    def axis_values(max_deg):
        n_steps = int(np.floor(max_deg / scanning_step_size_deg)) + 1
        values = [k * scanning_step_size_deg for k in range(n_steps)]
        # Ensure upper bound is visited as well.
        if values[-1] < max_deg:
            values.append(max_deg)
        return values

    az_values = axis_values(float(scanning_area_deg))
    el_values = axis_values(float(scanning_area_deg))

    for row, el in enumerate(el_values):
        if row % 2 == 0:
            sweep = az_values
        else:
            sweep = reversed(az_values)
        for az in sweep:
            scanning_list_azimuth_deg.append(float(az))
            scanning_list_elevation_deg.append(float(el))

    return scanning_list_azimuth_deg, scanning_list_elevation_deg

def raster_sector_scanning_centering_a_target(
    scanning_step_size_deg,
    scanning_area_deg,
    target_azimuth_deg,
    target_elevation_deg,
    num_steps_az=None,
    num_steps_el=None,
    max_positions=None):
    """
    Hedefe göre merkezlenmiş kare raster (serpentine).
    scanning_area_deg: kare kenar uzunluğu (°); toplam alan = scanning_area_deg x scanning_area_deg.

    num_steps_az / num_steps_el: verilirse azimuth/elevation adım SAYILARI doğrudan
        kullanılır (scanning_step_size_deg yok sayılır); ızgara tam sektörü kaplasın diye
        adım açıları area/(N-1) olarak hesaplanır. Kare olmayan (ör. 7x8) bir ızgarayla,
        bir tarama periyodundaki toplam huzme sayısı kare-olmayan bir frame sayısına
        (ör. 50) oturtulabilir.
    max_positions: verilirse serpentine listesi tam olarak bu uzunluğa kırpılır (son
        elevation satırı eksik/ragged kalır), böylece frame/periyot tam sayıya iner.
    """
    scanning_list_azimuth_deg=[]
    scanning_list_elevation_deg=[]

    def norm_az(az):
        return ((az%360)+360)%360
    def wrap_el(el):
        e=((el+90)%360)-90
        if e>90:
            e=180-e
        if e<-90:
            e=-180-e
        return e

    half_side = scanning_area_deg/2
    az_start = target_azimuth_deg - half_side
    el_start = target_elevation_deg - half_side

    if num_steps_az is not None and num_steps_el is not None:
        # Explicit grid shape: derive per-axis step so the grid spans the full sector.
        az_steps = max(1, int(num_steps_az))
        el_steps = max(1, int(num_steps_el))
        az_step = scanning_area_deg / max(1, az_steps - 1)
        el_step = scanning_area_deg / max(1, el_steps - 1)
    else:
        # Legacy: single step size, square grid.
        az_steps = int(np.floor(scanning_area_deg / scanning_step_size_deg)) + 1
        el_steps = int(np.floor(scanning_area_deg / scanning_step_size_deg)) + 1
        az_step = el_step = scanning_step_size_deg

    for i in range(el_steps):
        el = wrap_el(el_start + i*el_step)
        sweep = range(az_steps) if i % 2 == 0 else range(az_steps-1, -1, -1)
        for k in sweep:
            az = norm_az(az_start + k*az_step)
            scanning_list_azimuth_deg.append(az)
            scanning_list_elevation_deg.append(el)

    if max_positions is not None and 0 < max_positions < len(scanning_list_azimuth_deg):
        scanning_list_azimuth_deg = scanning_list_azimuth_deg[:max_positions]
        scanning_list_elevation_deg = scanning_list_elevation_deg[:max_positions]

    return scanning_list_azimuth_deg, scanning_list_elevation_deg


def conical_beam_az_el_deg(
    target_azimuth_deg,
    target_elevation_deg,
    scanning_area_deg,
    step_index,
    scanning_step_size_deg,
):
    """
    Beam direction (azimuth °, elevation °) on a small circle around the target direction.
    scanning_area_deg: tam koni genişliği (°); iç yarı-açı = scanning_area_deg / 2.
    Rotates 360° with step_index * scanning_step_size_deg.
    """
    if scanning_step_size_deg <= 0:
        raise ValueError("scanning_step_size_deg must be positive")
    if scanning_area_deg is None or scanning_area_deg <= 0:
        raise ValueError("scanning_area_deg must be positive")

    cone_half_angle_deg = float(scanning_area_deg) / 2.0
    radius_rad = np.deg2rad(cone_half_angle_deg)
    az0 = np.deg2rad(target_azimuth_deg)
    el0 = np.deg2rad(target_elevation_deg)

    cx = np.cos(el0) * np.cos(az0)
    cy = np.cos(el0) * np.sin(az0)
    cz = np.sin(el0)
    c = np.array([cx, cy, cz])

    t1 = np.array([-np.sin(az0), np.cos(az0), 0.0])
    t2 = np.array(
        [-np.sin(el0) * np.cos(az0), -np.sin(el0) * np.sin(az0), np.cos(el0)]
    )

    phi = np.deg2rad(step_index * scanning_step_size_deg)
    v = np.cos(radius_rad) * c + np.sin(radius_rad) * (
        np.cos(phi) * t1 + np.sin(phi) * t2
    )
    v = v / np.linalg.norm(v)
    x, y, z = v
    az = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    el = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return float(az), float(el)


def conical_scanning_centering_a_target(scanning_step_size_deg,
    scanning_area_deg,
    target_azimuth_deg=0, 
    target_elevation_deg=0):
    """
    Full one-revolution conical path around target (for plots / offline use).
    scanning_area_deg: tam koni genişliği (°); iç yarı-açı = scanning_area_deg / 2.
    """
    scanning_list_azimuth_deg=[]
    scanning_list_elevation_deg=[]

    if scanning_step_size_deg <= 0:
        raise ValueError("scanning_step_size_deg must be positive")
    if scanning_area_deg is None or scanning_area_deg <= 0:
        raise ValueError("scanning_area_deg must be positive")

    steps = int(np.ceil(360.0 / scanning_step_size_deg))
    for k in range(steps):
        az, el = conical_beam_az_el_deg(
            target_azimuth_deg,
            target_elevation_deg,
            scanning_area_deg,
            k,
            scanning_step_size_deg,
        )
        scanning_list_azimuth_deg.append(az)
        scanning_list_elevation_deg.append(el)

    return scanning_list_azimuth_deg, scanning_list_elevation_deg

def plot_scan_points_3d(scanning_list_azimuth_deg,
scanning_list_elevation_deg,
target_azimuth_deg=0,
target_elevation_deg=0, title="Scan on Unit Sphere"):
    def to_xyz(az_deg, el_deg):
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        return x, y, z

    xs, ys, zs = [], [], []
    for az_deg, el_deg in zip(scanning_list_azimuth_deg, scanning_list_elevation_deg):
        x, y, z = to_xyz(az_deg, el_deg)
        xs.append(x); ys.append(y); zs.append(z)

    tx, ty, tz = to_xyz(target_azimuth_deg, target_elevation_deg)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Draw unit sphere wireframe (light)
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(-np.pi/2, np.pi/2, 30)
    X = np.cos(v)[:, None] * np.cos(u)[None, :]
    Y = np.cos(v)[:, None] * np.sin(u)[None, :]
    Z = np.sin(v)[:, None] * np.ones_like(u)[None, :]
    ax.plot_wireframe(X, Y, Z, rstride=3, cstride=6, color='lightgray', linewidth=0.5, alpha=0.6)

    ax.plot(xs, ys, zs, color='tab:blue', linewidth=2, label='scan path')
    ax.scatter([tx], [ty], [tz], color='red', s=60, marker='x', label='target')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def _pattern_to_surface_vertices(pattern, gain_scale=1.0):
    az = np.asarray(pattern["Az_deg"], dtype=float)
    el = np.asarray(pattern["El_deg"], dtype=float)
    g  = np.asarray(pattern["Gain_linear"], dtype=float)

    az_unique = np.unique(az)
    el_unique = np.unique(el)
    az_idx = {val: i for i, val in enumerate(az_unique)}
    el_idx = {val: i for i, val in enumerate(el_unique)}
    G = np.full((len(el_unique), len(az_unique)), np.nan, dtype=float)
    for a, e, gv in zip(az, el, g):
        G[el_idx[e], az_idx[a]] = gv
    # Fill any missing by zero
    G = np.nan_to_num(G, nan=0.0)

    # Normalize for visibility
    if np.max(G) > 0:
        G = G / np.max(G)
    G = G * gain_scale

    AZ, EL = np.meshgrid(np.deg2rad(az_unique), np.deg2rad(el_unique))
    R = G
    X = R * np.cos(EL) * np.cos(AZ)
    Y = R * np.cos(EL) * np.sin(AZ)
    Z = R * np.sin(EL)
    return X, Y, Z, az_unique, el_unique

def plot_pattern_3d(pattern, gain_scale=1.0, title="Radiation pattern (3D)"):
    X, Y, Z, _, _ = _pattern_to_surface_vertices(pattern, gain_scale=gain_scale)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def _build_pattern_grid(pattern):
    az = np.asarray(pattern["Az_deg"], dtype=float)
    el = np.asarray(pattern["El_deg"], dtype=float)
    g_lin = np.asarray(pattern["Gain_linear"], dtype=float)
    az_unique = np.unique(az)
    el_unique = np.unique(el)
    az_idx = {val: i for i, val in enumerate(az_unique)}
    el_idx = {val: i for i, val in enumerate(el_unique)}
    G = np.full((len(el_unique), len(az_unique)), np.nan, dtype=float)
    for a, e, gv in zip(az, el, g_lin):
        G[el_idx[e], az_idx[a]] = gv
    G = np.nan_to_num(G, nan=0.0)
    return el_unique, az_unique, G

def build_gain_interpolator(pattern):
    el_unique, az_unique, G = _build_pattern_grid(pattern)
    interp = RegularGridInterpolator(
        (el_unique, az_unique), G, bounds_error=False, fill_value=0.0
    )
    return {"interp": interp, "el_grid": el_unique, "az_grid": az_unique}

def _norm_az_to_grid(az_deg, az_grid):
    az_grid_wrapped = (az_grid % 360.0 + 360.0) % 360.0
    val = (az_deg % 360.0 + 360.0) % 360.0
    d = np.abs(((az_grid_wrapped - val + 180.0) % 360.0) - 180.0)
    nearest = az_grid[int(np.argmin(d))]
    return nearest

def _vector_from_az_el(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z])

def _az_el_from_vector(v):
    x, y, z = v
    az = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    el = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return az, el

def _rot_z(az_rad):
    c = np.cos(az_rad); s = np.sin(az_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def _rot_y(el_rad):
    c = np.cos(el_rad); s = np.sin(el_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def compute_antenna_gain_towards_target(pattern_or_interp, pointing_az_deg, pointing_el_deg, target_az_deg, target_el_deg, return_dBi=False):
    if isinstance(pattern_or_interp, dict) and "interp" in pattern_or_interp:
        interp = pattern_or_interp["interp"]
        el_grid = pattern_or_interp["el_grid"]
        az_grid = pattern_or_interp["az_grid"]
    else:
        bundle = build_gain_interpolator(pattern_or_interp)
        interp = bundle["interp"]
        el_grid = bundle["el_grid"]
        az_grid = bundle["az_grid"]

    v_t = _vector_from_az_el(target_az_deg, target_el_deg)
    R_inv = _rot_y(pointing_el_deg * np.pi/180.0) @ _rot_z(-pointing_az_deg * np.pi/180.0)
    v_rel = R_inv @ v_t
    az_rel, el_rel = _az_el_from_vector(v_rel)

    az_rel = _norm_az_to_grid(az_rel, az_grid)
    el_rel = float(np.clip(el_rel, float(el_grid.min()), float(el_grid.max())))

    gain_linear = float(interp([[el_rel, az_rel]]))
    if return_dBi:
        with np.errstate(divide='ignore'):
            gain_dBi = 10.0 * np.log10(max(gain_linear, 1e-300))
        return gain_dBi
    return gain_linear

def simulate_received_power(pattern_or_interp, pointing_az_deg, pointing_el_deg, target_az_deg, target_el_deg, tx_power_watts=1.0):
    gain_linear = compute_antenna_gain_towards_target(
        pattern_or_interp, pointing_az_deg, pointing_el_deg, target_az_deg, target_el_deg, return_dBi=False
    )
    pr_w = tx_power_watts * gain_linear
    return gain_linear, pr_w

def animate_pattern_scan_on_sphere(pattern, scan_az_deg, scan_el_deg, interval_ms=50, gain_scale=1.0, title="Radiation pattern scanning (moving pattern)", align_boresight=False, azimuth_offset_deg=0.0, boresight_length=1.2):
    # Base surface in its intrinsic orientation
    X0, Y0, Z0, _, _ = _pattern_to_surface_vertices(pattern, gain_scale=gain_scale)

    def rot_z(az_rad):
        c = np.cos(az_rad); s = np.sin(az_rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    def rot_y(el_rad):
        c = np.cos(el_rad); s = np.sin(el_rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    # Pre-flatten base vertices for faster multiplication
    P0 = np.vstack([X0.ravel(), Y0.ravel(), Z0.ravel()])  # 3xN
    # If the intrinsic pattern boresight points at el=90 (Z+), align it to el=0, az=0 (X+)
    if align_boresight:
        R_align = rot_y(-np.pi/2)
        P0 = R_align @ P0

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-boresight_length, boresight_length]); ax.set_ylim([-boresight_length, boresight_length]); ax.set_zlim([-boresight_length, boresight_length])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)

    # Initial surface oriented to first scan angles
    az0 = np.deg2rad(scan_az_deg[0])
    el0 = np.deg2rad(scan_el_deg[0])
    R0 = rot_z(az0 + np.deg2rad(azimuth_offset_deg)) @ rot_y(-el0)
    P_init = R0 @ P0
    X_init = P_init[0, :].reshape(X0.shape)
    Y_init = P_init[1, :].reshape(Y0.shape)
    Z_init = P_init[2, :].reshape(Z0.shape)
    surf_container = {"surf": ax.plot_surface(X_init, Y_init, Z_init, cmap='viridis', edgecolor='none', alpha=0.85, antialiased=True)}
    # Boresight beam line (from origin to current pointing direction)
    beam_line, = ax.plot([0, 0], [0, 0], [0, 0], color='tab:red', linewidth=2, label='boresight')
    # Legend (include boresight)
    ax.legend(loc='upper left')

    def update(i):
        az_rad = np.deg2rad(scan_az_deg[i % len(scan_az_deg)])
        el_rad = np.deg2rad(scan_el_deg[i % len(scan_el_deg)])
        # Apply optional azimuth offset to align pattern with boresight line
        # Use Ry(-el) so that Rz(az) @ Ry(-el) applied to +X matches
        # the boresight unit vector [cos(el)cos(az), cos(el)sin(az), sin(el)].
        R = rot_z(az_rad + np.deg2rad(azimuth_offset_deg)) @ rot_y(-el_rad)
        P = R @ P0
        X = P[0, :].reshape(X0.shape)
        Y = P[1, :].reshape(Y0.shape)
        Z = P[2, :].reshape(Z0.shape)
        # Replace surface
        surf_container["surf"].remove()
        surf_container["surf"] = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85, antialiased=True)
        # Update boresight line to current pointing (unit vector)
        bx = boresight_length * (np.cos(el_rad) * np.cos(az_rad))
        by = boresight_length * (np.cos(el_rad) * np.sin(az_rad))
        bz = boresight_length * (np.sin(el_rad))
        beam_line.set_data([0, bx], [0, by])
        beam_line.set_3d_properties([0, bz])
        return (surf_container["surf"], beam_line)

    anim = animation.FuncAnimation(fig, update, frames=len(scan_az_deg), interval=interval_ms, blit=False, repeat=True)
    fig._anim = anim
    plt.tight_layout()
    plt.show()

def animate_pattern_and_gain(pattern, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg, interval_ms=30, gain_scale=1.0, azimuth_offset_deg=0.0, boresight_length=1.2, gain_x_axis="azimuth"):
    # Prepare surface vertices
    X0, Y0, Z0, _, _ = _pattern_to_surface_vertices(pattern, gain_scale=gain_scale)
    P0 = np.vstack([X0.ravel(), Y0.ravel(), Z0.ravel()])

    def rot_z(az_rad):
        c = np.cos(az_rad); s = np.sin(az_rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    def rot_y(el_rad):
        c = np.cos(el_rad); s = np.sin(el_rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


    # Build interpolator for gain computation
    interp = build_gain_interpolator(pattern)

    # Figure with two panes: left 3D, right 2D
    fig = plt.figure(figsize=(11, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax2d = fig.add_subplot(gs[0, 1])

    ax3d.set_box_aspect([1,1,1])
    ax3d.set_xlim([-boresight_length, boresight_length]); ax3d.set_ylim([-boresight_length, boresight_length]); ax3d.set_zlim([-boresight_length, boresight_length])
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title('Radiation pattern (3D)')

    # Initial orientation
    az0 = np.deg2rad(scan_az_deg[0])
    el0 = np.deg2rad(scan_el_deg[0])
    R0 = rot_z(az0 + np.deg2rad(azimuth_offset_deg)) @ rot_y(-el0)
    P_init = R0 @ P0
    X_init = P_init[0, :].reshape(X0.shape)
    Y_init = P_init[1, :].reshape(Y0.shape)
    Z_init = P_init[2, :].reshape(Z0.shape)
    surf_container = {"surf": ax3d.plot_surface(X_init, Y_init, Z_init, cmap='viridis', edgecolor='none', alpha=0.85, antialiased=True)}

    # Boresight line
    beam_line, = ax3d.plot([0, 0], [0, 0], [0, 0], color='tab:red', linewidth=2, label='boresight')
    
    # Receiver direction (fixed) as a line and marker (world coordinates)
    rx_vec = _vector_from_az_el(target_az_deg, target_el_deg)
    rx_x, rx_y, rx_z = boresight_length * float(rx_vec[0]), boresight_length * float(rx_vec[1]), boresight_length * float(rx_vec[2])
    ax3d.plot([0, rx_x], [0, rx_y], [0, rx_z], color='tab:green', linewidth=2, label='receiver')
    ax3d.scatter([rx_x], [rx_y], [rx_z], color='tab:green', s=40)
    # Project receiver onto XY plane for visual elevation cue
    ax3d.plot([rx_x, rx_x], [rx_y, rx_y], [0, rx_z], color='tab:green', linestyle=':', linewidth=1)
    # Label receiver az/el
    ax3d.text(rx_x, rx_y, rx_z, f"  RX az={target_az_deg:.1f}°, el={target_el_deg:.1f}°", color='tab:green')
    ax3d.legend(loc='upper left')

    # Gain plot setup
    ax2d.set_title('Gain toward receiver')
    # X-axis mode: 'azimuth' or 'step'
    use_step = str(gain_x_axis).lower() == "step"
    ax2d.set_xlabel('Scan step' if use_step else 'Azimuth (deg)')
    ax2d.set_ylabel('Gain (linear)')
    ax2d.grid(True, linestyle=":", alpha=0.6)
    az_full = np.asarray(scan_az_deg, dtype=float)
    el_full = np.asarray(scan_el_deg, dtype=float)
    gains_full = [
        compute_antenna_gain_towards_target(
            interp, az_val, el_val, target_az_deg, target_el_deg, return_dBi=False
        )
        for az_val, el_val in zip(az_full, el_full)
    ]
    x_full = np.arange(len(az_full)) if use_step else az_full
    gain_curve, = ax2d.plot(x_full, gains_full, color='tab:blue', lw=2, label='Gain (linear)')
    current_point, = ax2d.plot([x_full[0]], [gains_full[0]], marker='o', color='tab:red', ms=6, label='Current')
    ax2d.legend(loc='best')

    def update(i):
        az_rad = np.deg2rad(scan_az_deg[i % len(scan_az_deg)])
        el_rad = np.deg2rad(scan_el_deg[i % len(scan_el_deg)])
        R = rot_z(az_rad + np.deg2rad(azimuth_offset_deg)) @ rot_y(-el_rad)
        P = R @ P0
        X = P[0, :].reshape(X0.shape)
        Y = P[1, :].reshape(Y0.shape)
        Z = P[2, :].reshape(Z0.shape)
        # Replace surface
        surf_container["surf"].remove()
        surf_container["surf"] = ax3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85, antialiased=True)
        # Update boresight line
        bx = boresight_length * (np.cos(el_rad) * np.cos(az_rad))
        by = boresight_length * (np.cos(el_rad) * np.sin(az_rad))
        bz = boresight_length * (np.sin(el_rad))
        beam_line.set_data([0, bx], [0, by])
        beam_line.set_3d_properties([0, bz])

        # Update red marker on precomputed curve
        az_deg_now = float(np.degrees(az_rad))
        idx_near = int(np.argmin(np.abs(az_full - az_deg_now)))
        x_now = idx_near if use_step else az_full[idx_near]
        current_point.set_data([x_now], [gains_full[idx_near]])
        return (surf_container["surf"], beam_line, current_point)

    anim = animation.FuncAnimation(fig, update, frames=len(scan_az_deg), interval=interval_ms, blit=False, repeat=True)
    fig._anim = anim
    fig.tight_layout()
    plt.show()

def antenna_gain_calculator(pattern, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg):
    az_vals = np.asarray(pattern["Az_deg"], dtype=float)
    el_vals = np.asarray(pattern["El_deg"], dtype=float)
    gain_vals = np.asarray(pattern["Gain_linear"], dtype=float)

    rel_el = float(target_el_deg) - float(scan_el_deg)
    rel_az = (float(target_az_deg) - float(scan_az_deg)) % 360.0
    rel_az = (rel_az + 180.0) % 360.0 - 180.0
    rel_el = max(-90.0, min(90.0, rel_el))
    
    az_diff = np.abs((az_vals - rel_az + 180.0) % 360.0 - 180.0)
    el_diff = np.abs(el_vals - rel_el)
    total_diff = np.hypot(az_diff, el_diff)

    idx_sorted = np.argsort(total_diff)
    i1, i2 = idx_sorted[:2]
    d1, d2 = total_diff[i1], total_diff[i2]
    g1, g2 = gain_vals[i1], gain_vals[i2]

    # handle identical distance case
    if d1 == 0:
        gain = g1
    else:
        gain = (g1 * d2 + g2 * d1) / (d1 + d2)
    return gain

if __name__ == "__main__":
    # 1) Load radiation pattern
    pattern = load_radiation_pattern("parabolic_reflector")

    # 2) Set target direction in world coordinates (az/el in degrees)
    target_az_deg = 30.0
    target_el_deg = -5.0

    
    # Circular Scanning
    scan_az_deg_list, scan_el_deg_list = circular_scanning(5, 0, 0)
    gain_list_circular = []
    for step_idx, (scan_az_deg, scan_el_deg) in enumerate(zip(scan_az_deg_list, scan_el_deg_list)):
        gain = antenna_gain_calculator(pattern, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg)
        gain_list_circular.append(gain)

    # Unidirectional Sector Scanning
    scan_az_deg_list, scan_el_deg_list = unidirectional_sector_scanning(5, 40, 0, 0)
    gain_list_unidirectional_sector = []
    for step_idx, (scan_az_deg, scan_el_deg) in enumerate(zip(scan_az_deg_list, scan_el_deg_list)):
        gain = antenna_gain_calculator(pattern, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg)
        gain_list_unidirectional_sector.append(gain)
        
    # Bidirectional Sector Scanning
    scan_az_deg_list, scan_el_deg_list = bidirectional_sector_scanning(5, 40, 0, 0)
    gain_list_bidirectional_sector = []
    for step_idx, (scan_az_deg, scan_el_deg) in enumerate(zip(scan_az_deg_list, scan_el_deg_list)):
        gain = antenna_gain_calculator(pattern, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg)
        gain_list_bidirectional_sector.append(gain)
        
    # Raster Scanning
    scan_az_deg_list, scan_el_deg_list = raster_scanning(10, 90, 0, 0)
    gain_list_raster = []
    for step_idx, (scan_az_deg, scan_el_deg) in enumerate(zip(scan_az_deg_list, scan_el_deg_list)):
        gain = antenna_gain_calculator(pattern, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg)
        gain_list_raster.append(gain)
        
    # Raster Sector Scanning Centering a Target
    scan_az_deg_list, scan_el_deg_list = raster_sector_scanning_centering_a_target(5, 40, 0, 0)
    gain_list_raster_sector_target = []
    for step_idx, (scan_az_deg, scan_el_deg) in enumerate(zip(scan_az_deg_list, scan_el_deg_list)):
        gain = antenna_gain_calculator(pattern, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg)
        gain_list_raster_sector_target.append(gain)
        
    # Conical Scanning Centering a Target
    scan_az_deg_list, scan_el_deg_list = conical_scanning_centering_a_target(10, 30, 0, 0)
    gain_list_conical_target = []
    for step_idx, (scan_az_deg, scan_el_deg) in enumerate(zip(scan_az_deg_list, scan_el_deg_list)):
        gain = antenna_gain_calculator(pattern, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg)
        gain_list_conical_target.append(gain)
    
    # Extend all gain lists to equal length by cycling shorter patterns
    all_gain_lists = [
        gain_list_circular,
        gain_list_unidirectional_sector,
        gain_list_bidirectional_sector,
        gain_list_raster,
        gain_list_raster_sector_target,
        gain_list_conical_target
    ]
    
    # Find maximum length
    max_length = max(len(gain_list) for gain_list in all_gain_lists)
    
    # Extend each list by cycling if shorter

    gain_list_circular = (gain_list_circular * ((max_length // len(gain_list_circular)) + 1))[:max_length]

    gain_list_unidirectional_sector = (gain_list_unidirectional_sector * ((max_length // len(gain_list_unidirectional_sector)) + 1))[:max_length]

    gain_list_bidirectional_sector = (gain_list_bidirectional_sector * ((max_length // len(gain_list_bidirectional_sector)) + 1))[:max_length]


    gain_list_raster = (gain_list_raster * ((max_length // len(gain_list_raster)) + 1))[:max_length]

    gain_list_raster_sector_target = (gain_list_raster_sector_target * ((max_length // len(gain_list_raster_sector_target)) + 1))[:max_length]

    gain_list_conical_target = (gain_list_conical_target * ((max_length // len(gain_list_conical_target)) + 1))[:max_length]
    
    # Convert to numpy arrays for easier manipulation
    gain_list_circular = np.array(gain_list_circular)
    gain_list_unidirectional_sector = np.array(gain_list_unidirectional_sector)
    gain_list_bidirectional_sector = np.array(gain_list_bidirectional_sector)
    gain_list_raster = np.array(gain_list_raster)
    gain_list_raster_sector_target = np.array(gain_list_raster_sector_target)
    gain_list_conical_target = np.array(gain_list_conical_target)
    
    # Add noise to each gain list separately
    # Noise power levels in dB
    noise_power_levels_db = [-10, -5, 0, 5]
    
    # Save noisy gain lists
    output_dir = "gain_lists_noisy"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 200 versions of each noisy gain list for each noise power level
    num_versions = 500
    
    # Define scanning type names for file naming (simple names)
    scanning_types = {
        'circular': gain_list_circular,
        'unidirectional_sector': gain_list_unidirectional_sector,
        'bidirectional_sector': gain_list_bidirectional_sector,
        'raster': gain_list_raster,
        'raster_sector_target': gain_list_raster_sector_target,
        'conical_target': gain_list_conical_target
    }
    
    # Loop through each noise power level
    for noise_power_db in noise_power_levels_db:
        # Convert noise power from dB to linear scale (variance)
        noise_power_linear = 10 ** (noise_power_db / 10)
        noise_std = np.sqrt(noise_power_linear)
        
        # Format noise power for filename (remove decimal, handle negative)
        noise_power_str = f"{int(noise_power_db)}db"
        if noise_power_db >= 0:
            noise_power_str = f"+{int(noise_power_db)}db"
        
        # Generate 200 versions for this noise power level
        for version_idx in range(num_versions):
            for scan_type, gain_list in scanning_types.items():
                # Generate new noise for this version
                noise = np.random.normal(0, noise_std, size=gain_list.shape)
                
                # Add noise to gain list
                gain_list_noisy = gain_list + noise
                
                # Save with simple naming: {type}_{noise_power}db_{index}.npy
                filename = f"{scan_type}_{noise_power_str}_{version_idx}.npy"
                filepath = os.path.join(output_dir, filename)
                np.save(filepath, gain_list_noisy)
    
