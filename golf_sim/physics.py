"""2D projectile motion with quadratic air drag for a golf ball."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665  # m/s^2
BALL_MASS_KG = 0.04593
BALL_RADIUS_M = 0.021335
BALL_AREA_M2 = np.pi * BALL_RADIUS_M ** 2
AIR_DENSITY = 1.225  # kg/m^3, sea level
DRAG_COEFFICIENT = 0.25
MPS_TO_MPH = 2.236936
M_TO_YARDS = 1.093613
YARDS_TO_M = 1.0 / M_TO_YARDS


@dataclass
class ShotResult:
    ball_speed_mph: float
    launch_angle_deg: float
    apex_yards: float
    carry_yards: float
    points_yards: np.ndarray  # shape (N, 2) — x carry, y height
    # --- added: A.O.A. + roll/total ---
    aoa_deg: float = 0.0
    aoa_label: str = "--"
    roll_yards: float = 0.0
    total_yards: float = 0.0
    landing_speed_mps: float = 0.0
    landing_angle_deg: float = 0.0


def format_aoa(aoa_deg: float | None) -> str:
    """Dashboard string, e.g. '-2.5° Down' or '+3.0° Up'."""
    if aoa_deg is None:
        return "--"
    if abs(aoa_deg) < 0.15:
        return f"{aoa_deg:+.1f}° Level"
    direction = "Up" if aoa_deg > 0 else "Down"
    return f"{aoa_deg:+.1f}° {direction}"


def estimated_roll_yards(
    launch_angle_deg: float,
    landing_speed_mps: float,
    landing_angle_deg: float,
) -> float:
    """Fairway bounce-and-roll from landing speed and impact steepness.

    Total Distance = Carry + this roll. Steep landings (wedges) roll less;
    shallow landings with leftover horizontal speed roll more.
    """
    landing_speed_mps = max(0.0, float(landing_speed_mps))
    impact_steepness = abs(float(landing_angle_deg))  # 90° = vertical drop
    theta_land = np.deg2rad(impact_steepness)
    vx_land = landing_speed_mps * np.cos(theta_land)

    mu_fairway = 0.18
    bounce = float(np.clip(1.15 - impact_steepness / 55.0, 0.12, 1.0))
    # Energy-based slide, scaled down to typical turf (most energy is lost at bounce).
    roll_m = (vx_land ** 2) / (2.0 * mu_fairway * G) * bounce * 0.055
    launch_factor = float(np.clip(1.0 - (float(launch_angle_deg) - 12.0) / 42.0, 0.08, 1.0))
    roll_m *= launch_factor
    return max(0.0, roll_m * M_TO_YARDS)


def launch_velocity_components(speed_mps: float, launch_angle_deg: float) -> tuple[float, float]:
    theta = np.deg2rad(launch_angle_deg)
    return float(speed_mps * np.cos(theta)), float(speed_mps * np.sin(theta))


def simulate_flight(
    speed_mps: float,
    launch_angle_deg: float,
    *,
    aoa_deg: float | None = None,
    dt: float = 0.008,
    max_time_s: float = 12.0,
    tee_height_m: float = 0.03,
    cd: float = DRAG_COEFFICIENT,
) -> ShotResult:
    """Integrate quadratic-drag projectile motion until the ball returns to ground.

    Drag force:  F_d = 0.5 * Cd * rho * A * |v|^2  opposite velocity.
    """
    speed_mps = max(0.0, float(speed_mps))
    vx, vy = launch_velocity_components(speed_mps, launch_angle_deg)
    x = 0.0
    y = max(0.0, tee_height_m)

    xs = [x]
    ys = [y]
    apex_m = y
    t = 0.0
    mass = BALL_MASS_KG

    while t < max_time_s:
        speed = float(np.hypot(vx, vy))
        if speed < 1e-6:
            break

        drag_acc = 0.5 * cd * AIR_DENSITY * BALL_AREA_M2 * speed / mass
        ax = -drag_acc * vx
        ay = -G - drag_acc * vy

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        if y < 0.0:
            # Linear interpolation back to ground contact for carry.
            prev_x, prev_y = xs[-1], ys[-1]
            frac = prev_y / (prev_y - y) if (prev_y - y) != 0 else 1.0
            x = prev_x + frac * (x - prev_x)
            y = 0.0
            xs.append(x)
            ys.append(y)
            break

        xs.append(x)
        ys.append(y)
        if y > apex_m:
            apex_m = y

    points_m = np.column_stack((np.array(xs), np.array(ys)))
    points_yd = points_m * M_TO_YARDS
    carry_m = float(points_m[-1, 0]) if len(points_m) else 0.0
    carry_yd = max(0.0, carry_m * M_TO_YARDS)

    landing_speed = float(np.hypot(vx, vy))
    landing_angle = float(np.rad2deg(np.arctan2(vy, vx)))
    roll_yd = estimated_roll_yards(launch_angle_deg, landing_speed, landing_angle)
    roll_yd = min(roll_yd, carry_yd * 0.55)
    aoa_val = 0.0 if aoa_deg is None else float(aoa_deg)

    return ShotResult(
        ball_speed_mph=speed_mps * MPS_TO_MPH,
        launch_angle_deg=float(launch_angle_deg),
        apex_yards=apex_m * M_TO_YARDS,
        carry_yards=carry_yd,
        points_yards=points_yd,
        aoa_deg=aoa_val,
        aoa_label=format_aoa(None if aoa_deg is None else aoa_val),
        roll_yards=roll_yd,
        total_yards=carry_yd + roll_yd,
        landing_speed_mps=landing_speed,
        landing_angle_deg=landing_angle,
    )


def kinematics_from_centroids(
    centroids_px: list[tuple[float, float]],
    timestamps_s: list[float],
    meters_per_pixel: float,
) -> tuple[float, float]:
    """Ball speed (m/s) and launch angle (deg) from image-space centroids.

    Image y grows downward, so world-up is -dy.
    Uses the first-to-last valid pair with a robust median of frame-to-frame speeds.
    """
    if len(centroids_px) < 2 or len(timestamps_s) < 2:
        return 0.0, 0.0

    meters_per_pixel = max(1e-6, float(meters_per_pixel))
    speeds = []
    angles = []

    for i in range(1, min(len(centroids_px), len(timestamps_s))):
        dt = timestamps_s[i] - timestamps_s[i - 1]
        if dt <= 1e-4:
            continue
        x0, y0 = centroids_px[i - 1]
        x1, y1 = centroids_px[i]
        dx_m = (x1 - x0) * meters_per_pixel
        dy_m = -(y1 - y0) * meters_per_pixel  # image down → world up
        vx, vy = dx_m / dt, dy_m / dt
        speed = float(np.hypot(vx, vy))
        if speed < 0.05:
            continue
        speeds.append(speed)
        angles.append(float(np.rad2deg(np.arctan2(vy, vx))))

    if not speeds:
        return 0.0, 0.0

    return float(np.median(speeds)), float(np.median(angles))
