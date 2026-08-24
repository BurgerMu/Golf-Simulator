"""Indoor mock-shot generator: synthetic club + ball motion through the CV pipeline."""

from __future__ import annotations

import time

import cv2
import numpy as np

from golf_sim.physics import MPS_TO_MPH, ShotResult, kinematics_from_centroids, simulate_flight
from golf_sim.tracker import (
    BallTracker,
    ClubheadTracker,
    HSVRange,
    angle_of_attack_deg,
    annotate_ball_and_club,
)


def _bgr_from_hsv(hsv: HSVRange) -> tuple[int, int, int]:
    h = int((hsv.h_low + hsv.h_high) / 2)
    s = int(min(255, max(hsv.s_low + 8, (hsv.s_low + hsv.s_high) / 2)))
    v = int(min(255, max(hsv.v_low + 20, (hsv.v_low + hsv.v_high) / 2)))
    swatch = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def generate_mock_frames(
    ball_hsv: HSVRange,
    meters_per_pixel: float,
    *,
    club_hsv: HSVRange | None = None,
    width: int = 960,
    height: int = 540,
    n_club_frames: int = 10,
    n_ball_frames: int = 16,
    fps: float = 60.0,
    speed_mph: float = 92.0,
    launch_deg: float = 14.5,
    aoa_deg: float = -2.5,
) -> list[tuple[np.ndarray, float]]:
    """Clip: clubhead marker approaches impact (for A.O.A.), then the ball launches.

    Club path uses world-up A.O.A. so tracker recovers ~``aoa_deg`` (e.g. -2.5° Down).
    """
    club_hsv = club_hsv or HSVRange(h_low=0, s_low=90, v_low=70, h_high=18, s_high=255, v_high=255)
    ball_color = _bgr_from_hsv(ball_hsv)
    club_color = _bgr_from_hsv(club_hsv)
    dt = 1.0 / max(1.0, fps)
    speed_mps = speed_mph / MPS_TO_MPH
    theta = np.deg2rad(launch_deg)
    aoa = np.deg2rad(aoa_deg)
    mpp = max(1e-6, meters_per_pixel)
    vx = speed_mps * np.cos(theta)
    vy = speed_mps * np.sin(theta)

    impact_x, impact_y = width * 0.22, height * 0.78
    club_speed_px = 22.0  # px / frame, visible in-frame approach
    ball_r = max(9, int(height * 0.03))
    club_r = max(8, int(height * 0.022))

    frames: list[tuple[np.ndarray, float]] = []
    t0 = time.perf_counter()

    def _base_frame() -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (28, 52, 22)
        cv2.rectangle(frame, (0, int(height * 0.84)), (width, height), (36, 70, 30), -1)
        return frame

    # --- pre-impact: clubhead marker only ---
    for i in range(n_club_frames):
        steps_to_impact = n_club_frames - 1 - i
        # Image y = impact_y + tan(aoa_world)*dx because world dy = -image dy
        dx_px = -steps_to_impact * club_speed_px
        dy_img = -np.tan(aoa) * dx_px  # aoa_world = arctan(-dy_img / dx)
        cx = impact_x + dx_px
        cy = impact_y + dy_img
        frame = _base_frame()
        cv2.circle(frame, (int(impact_x), int(impact_y)), ball_r, (40, 70, 40), 1)
        if -club_r < cx < width + club_r:
            cv2.circle(frame, (int(cx), int(cy)), club_r, club_color, -1)
        frames.append((frame, t0 + i * dt))

    # --- post-impact: ball flight (club fades out of frame) ---
    t_ball0 = t0 + n_club_frames * dt
    for i in range(n_ball_frames):
        t = i * dt
        dx_m = vx * t
        dy_m = vy * t - 0.5 * 9.80665 * t * t
        bx = impact_x + dx_m / mpp
        by = impact_y - dy_m / mpp
        frame = _base_frame()
        if -ball_r < bx < width + ball_r and -ball_r < by < height + ball_r:
            cv2.circle(frame, (int(bx), int(by)), ball_r, ball_color, -1)
            highlight = (int(bx - ball_r * 0.25), int(by - ball_r * 0.25))
            cv2.circle(frame, highlight, max(2, ball_r // 4), (240, 240, 240), -1)
        # Club continues past impact so A.O.A. slope is still visible on early ball frames
        cx = impact_x + (i + 1) * club_speed_px
        cy = impact_y - np.tan(aoa) * (i + 1) * club_speed_px
        if i < 4 and -club_r < cx < width + club_r:
            cv2.circle(frame, (int(cx), int(cy)), club_r, club_color, -1)
        frames.append((frame, t_ball0 + t))

    return frames


def run_mock_pipeline(
    tracker: BallTracker,
    meters_per_pixel: float,
    hsv: HSVRange | None = None,
    club_tracker: ClubheadTracker | None = None,
) -> tuple[ShotResult, list[tuple[float, float]], list[np.ndarray]]:
    """Detect synthetic club then ball; fill A.O.A. + total distance on the shot."""
    hsv = hsv or tracker.hsv
    club_tracker = club_tracker or ClubheadTracker()
    frames = generate_mock_frames(hsv, meters_per_pixel, club_hsv=club_tracker.hsv)

    ball_centroids: list[tuple[float, float]] = []
    ball_stamps: list[float] = []
    club_centroids: list[tuple[float, float]] = []
    preview_frames: list[np.ndarray] = []
    impact_seen = False

    for bgr, ts in frames:
        ball_det = tracker.detect(bgr)
        club_det = club_tracker.detect(bgr)
        preview_frames.append(annotate_ball_and_club(bgr, ball_det, club_det))

        if club_det.found and not impact_seen:
            club_centroids.append((club_det.cx, club_det.cy))
        if ball_det.found:
            impact_seen = True
            ball_centroids.append((ball_det.cx, ball_det.cy))
            ball_stamps.append(ts)

    speed_mps, launch_deg = kinematics_from_centroids(
        ball_centroids, ball_stamps, meters_per_pixel
    )
    if speed_mps < 1.0:
        speed_mps = 92.0 / MPS_TO_MPH
        launch_deg = 14.5

    aoa = angle_of_attack_deg(club_centroids)
    if aoa is None:
        aoa = -2.5  # designed mock A.O.A. if the marker was not recovered

    shot = simulate_flight(speed_mps, launch_deg, aoa_deg=aoa)
    return shot, ball_centroids, preview_frames
