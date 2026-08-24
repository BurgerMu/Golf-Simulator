"""HSV golf-ball detector using blur + contour tracking."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HSVRange:
    h_low: int = 0
    s_low: int = 0
    v_low: int = 180
    h_high: int = 180
    s_high: int = 60
    v_high: int = 255

    def lower(self) -> np.ndarray:
        return np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)

    def upper(self) -> np.ndarray:
        return np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)


@dataclass
class Detection:
    found: bool
    cx: float
    cy: float
    radius: float
    mask: np.ndarray


class BallTracker:
    def __init__(self, hsv: HSVRange | None = None, min_area: int = 80) -> None:
        self.hsv = hsv or HSVRange()
        self.min_area = min_area

    def detect(self, bgr: np.ndarray) -> Detection:
        blurred = cv2.GaussianBlur(bgr, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv.lower(), self.hsv.upper())
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            peri = cv2.arcLength(contour, True)
            if peri <= 0:
                continue
            circularity = 4.0 * np.pi * area / (peri * peri)
            if circularity < 0.55:
                continue
            score = area * circularity
            if score > best_score:
                best_score = score
                best = contour

        if best is None:
            return Detection(False, 0.0, 0.0, 0.0, mask)

        (cx, cy), radius = cv2.minEnclosingCircle(best)
        return Detection(True, float(cx), float(cy), float(radius), mask)

    def annotate(self, bgr: np.ndarray, det: Detection) -> np.ndarray:
        out = bgr.copy()
        if det.found:
            center = (int(det.cx), int(det.cy))
            cv2.circle(out, center, max(2, int(det.radius)), (0, 220, 80), 2)
            cv2.drawMarker(out, center, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                out,
                f"BALL ({int(det.cx)}, {int(det.cy)})",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                out,
                "No ball detected",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 40, 255),
                2,
                cv2.LINE_AA,
            )
        return out


# Default clubhead sticker: saturated red/orange, distinct from a white ball.
CLUB_HSV_DEFAULT = HSVRange(h_low=0, s_low=90, v_low=70, h_high=18, s_high=255, v_high=255)


class ClubheadTracker:
    """HSV blob tracker for a clubhead marker (tape/sticker) — less circularity-strict."""

    def __init__(self, hsv: HSVRange | None = None, min_area: int = 40) -> None:
        self.hsv = hsv or HSVRange(
            h_low=CLUB_HSV_DEFAULT.h_low,
            s_low=CLUB_HSV_DEFAULT.s_low,
            v_low=CLUB_HSV_DEFAULT.v_low,
            h_high=CLUB_HSV_DEFAULT.h_high,
            s_high=CLUB_HSV_DEFAULT.s_high,
            v_high=CLUB_HSV_DEFAULT.v_high,
        )
        self.min_area = min_area

    def detect(self, bgr: np.ndarray) -> Detection:
        blurred = cv2.GaussianBlur(bgr, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv.lower(), self.hsv.upper())
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if area > best_area:
                best_area = area
                best = contour

        if best is None:
            return Detection(False, 0.0, 0.0, 0.0, mask)

        (cx, cy), radius = cv2.minEnclosingCircle(best)
        return Detection(True, float(cx), float(cy), float(radius), mask)


def angle_of_attack_deg(club_centroids_px: list[tuple[float, float]]) -> float | None:
    """A.O.A. from clubhead marker slope on frames leading into impact.

    Image y grows downward. World-up A.O.A. uses:
        AOA = arctan( -(y2 - y1) / (x2 - x1) )
    Positive = club rising (Up), negative = club descending (Down).
    """
    if len(club_centroids_px) < 2:
        return None

    samples = club_centroids_px[-16:]
    angles: list[float] = []
    for i in range(1, len(samples)):
        x0, y0 = samples[i - 1]
        x1, y1 = samples[i]
        dx = x1 - x0
        dy_img = y1 - y0
        if abs(dx) < 0.4 and abs(dy_img) < 0.4:
            continue
        dy_world = -dy_img
        angles.append(float(np.rad2deg(np.arctan2(dy_world, dx))))

    if angles:
        return float(np.median(angles))

    x0, y0 = samples[0]
    x1, y1 = samples[-1]
    return float(np.rad2deg(np.arctan2(-(y1 - y0), (x1 - x0))))


def annotate_ball_and_club(
    bgr: np.ndarray,
    ball: Detection,
    club: Detection,
) -> np.ndarray:
    """Overlay ball (green) and clubhead marker (orange) on the live frame."""
    tracker = BallTracker()
    out = tracker.annotate(bgr, ball)
    if club.found:
        center = (int(club.cx), int(club.cy))
        cv2.circle(out, center, max(3, int(club.radius)), (0, 140, 255), 2)
        cv2.drawMarker(out, center, (0, 200, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(
            out,
            "CLUB",
            (center[0] + 8, center[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def combine_masks(ball_mask: np.ndarray, club_mask: np.ndarray) -> np.ndarray:
    """Green = ball HSV, red = clubhead HSV (for the mask preview panel)."""
    combo = np.zeros((*ball_mask.shape, 3), dtype=np.uint8)
    combo[:, :, 1] = ball_mask
    combo[:, :, 2] = club_mask
    return combo
