"""PyQt5 dashboard: live camera, HSV calibration, metrics, trajectory canvas."""

from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from golf_sim.camera import Camera
from golf_sim.mock_shot import run_mock_pipeline
from golf_sim.physics import ShotResult, kinematics_from_centroids, simulate_flight
from golf_sim.tracker import (
    CLUB_HSV_DEFAULT,
    BallTracker,
    ClubheadTracker,
    HSVRange,
    angle_of_attack_deg,
    annotate_ball_and_club,
    combine_masks,
)


STYLESHEET = """
QWidget {
    background: #0d1410;
    color: #e8f0e6;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2a4a34;
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: #9ad0a8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background: #1f6b3a;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 700;
    font-size: 13px;
}
QPushButton:hover { background: #278a49; }
QPushButton:pressed { background: #16532c; }
QPushButton#danger {
    background: #5a2b2b;
}
QPushButton#danger:hover { background: #7a3838; }
QPushButton#ghost {
    background: #1a2a20;
    border: 1px solid #2a4a34;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1c2e24;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #4ade80;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QDoubleSpinBox {
    background: #132018;
    border: 1px solid #2a4a34;
    border-radius: 6px;
    padding: 6px;
    color: #f4fff4;
}
QLabel#status {
    color: #8fb89a;
    font-size: 12px;
}
QTabWidget::pane {
    border: 1px solid #2a4a34;
    border-radius: 6px;
    background: #101a14;
}
QTabBar::tab {
    background: #1a2a20;
    color: #9ad0a8;
    padding: 6px 12px;
    border: 1px solid #2a4a34;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 3px;
}
QTabBar::tab:selected {
    background: #1f6b3a;
    color: #ffffff;
}
"""


class MetricCard(QFrame):
    def __init__(self, title: str, unit: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setStyleSheet(
            """
            QFrame#metricCard {
                background: #132018;
                border: 1px solid #2f5a3d;
                border-radius: 12px;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        cap = QLabel(title.upper())
        cap.setStyleSheet("color: #7dba8e; font-size: 11px; letter-spacing: 1px; font-weight: 700;")
        self.value = QLabel("--")
        self.value.setStyleSheet("color: #f7fff8; font-size: 22px; font-weight: 800;")
        self.value.setWordWrap(True)
        unit_lbl = QLabel(unit)
        unit_lbl.setStyleSheet("color: #a9cbb4; font-size: 12px;")
        layout.addWidget(cap)
        layout.addWidget(self.value)
        layout.addWidget(unit_lbl)

    def set_value(self, text: str) -> None:
        self.value.setText(text)


class TrajectoryCanvas(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.points = np.empty((0, 2))
        self.apex_yards = 0.0
        self.carry_yards = 0.0
        self.total_yards = 0.0
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_shot(self, shot: ShotResult | None) -> None:
        if shot is None:
            self.points = np.empty((0, 2))
            self.apex_yards = 0.0
            self.carry_yards = 0.0
            self.total_yards = 0.0
        else:
            self.points = shot.points_yards
            self.apex_yards = shot.apex_yards
            self.carry_yards = shot.carry_yards
            self.total_yards = shot.total_yards
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#101c14"))

        margin_l, margin_r, margin_t, margin_b = 48, 18, 18, 36
        plot_w = max(1, w - margin_l - margin_r)
        plot_h = max(1, h - margin_t - margin_b)

        max_x = max(40.0, max(self.carry_yards, self.total_yards) * 1.08 if self.carry_yards or self.total_yards else 40.0)
        max_y = max(15.0, self.apex_yards * 1.35 if self.apex_yards else 15.0)

        def to_view(x_yd: float, y_yd: float) -> QPointF:
            px = margin_l + (x_yd / max_x) * plot_w
            py = margin_t + plot_h - (y_yd / max_y) * plot_h
            return QPointF(px, py)

        # Ground + grid
        painter.setPen(QPen(QColor("#1e3a28"), 1))
        for i in range(1, 5):
            y = margin_t + plot_h * i / 5
            painter.drawLine(margin_l, int(y), w - margin_r, int(y))
            x = margin_l + plot_w * i / 5
            painter.drawLine(int(x), margin_t, int(x), margin_t + plot_h)

        painter.setPen(QPen(QColor("#4ade80"), 2))
        ground_y = margin_t + plot_h
        painter.drawLine(margin_l, ground_y, w - margin_r, ground_y)

        painter.setPen(QColor("#8fb89a"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(8, ground_y + 4, "0")
        painter.drawText(margin_l, 14, f"{max_y:.0f} yd")
        painter.drawText(w - 70, ground_y + 18, f"{max_x:.0f} yd")
        painter.drawText(int(w / 2 - 40), int(h - 8), "Carry (yards)")

        if len(self.points) < 2:
            painter.setPen(QColor("#6f8f78"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Trajectory will appear after a shot")
            return

        path = QPainterPath()
        first = to_view(float(self.points[0, 0]), float(self.points[0, 1]))
        path.moveTo(first)
        for x, y in self.points[1:]:
            path.lineTo(to_view(float(x), float(y)))

        fill = QPainterPath(path)
        fill.lineTo(to_view(float(self.points[-1, 0]), 0.0))
        fill.lineTo(to_view(0.0, 0.0))
        fill.closeSubpath()
        grad = QLinearGradient(0, margin_t, 0, ground_y)
        grad.setColorAt(0.0, QColor(74, 222, 128, 70))
        grad.setColorAt(1.0, QColor(74, 222, 128, 10))
        painter.fillPath(fill, grad)

        painter.setPen(QPen(QColor("#4ade80"), 3))
        painter.drawPath(path)

        apex_idx = int(np.argmax(self.points[:, 1]))
        apex_pt = to_view(float(self.points[apex_idx, 0]), float(self.points[apex_idx, 1]))
        painter.setBrush(QColor("#facc15"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(apex_pt, 5, 5)
        painter.setPen(QColor("#fde68a"))
        painter.drawText(apex_pt + QPointF(8, -6), f"Apex {self.apex_yards:.1f} yd")

        landing = to_view(self.carry_yards, 0.0)
        painter.setBrush(QColor("#fb7185"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(landing, 5, 5)

        # Roll segment along the ground: carry → total distance
        if self.total_yards > self.carry_yards + 0.3:
            roll_end = to_view(self.total_yards, 0.0)
            pen = QPen(QColor("#fbbf24"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(landing, roll_end)
            painter.setBrush(QColor("#fbbf24"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(roll_end, 5, 5)
            painter.setPen(QColor("#fde68a"))
            painter.drawText(roll_end + QPointF(-36, -8), f"Total {self.total_yards:.0f}")


class VideoPanel(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #0a100c; border: 1px solid #2a4a34; border-radius: 10px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.show_placeholder("Camera paused")

    def show_placeholder(self, text: str = "Camera paused") -> None:
        pix = QPixmap(self.width() or 640, self.height() or 360)
        pix.fill(QColor("#0a100c"))
        p = QPainter(pix)
        p.setPen(QColor("#6f8f78"))
        p.setFont(QFont("Segoe UI", 14))
        p.drawText(pix.rect(), Qt.AlignCenter, text)
        p.end()
        self.setPixmap(pix)

    def show_bgr(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pix)


class GolfSimulatorWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DIY All-in-One Golf Simulator")
        self.resize(1280, 900)
        self.setStyleSheet(STYLESHEET)

        self.camera = Camera(0)
        self.tracker = BallTracker(HSVRange())
        self.club_tracker = ClubheadTracker()
        self.paused = True
        self.live_centroids: list[tuple[float, float]] = []
        self.live_stamps: list[float] = []
        self.live_club_centroids: list[tuple[float, float]] = []
        self.aoa_locked: float | None = None
        self.miss_frames = 0
        self.shot_locked = False
        self._mock_preview: list[np.ndarray] = []
        self._mock_i = 0

        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(16)

        self._try_open_camera()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        title = QLabel("DIY ALL-IN-ONE GOLF SIMULATOR")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #ecfdf3; letter-spacing: 1px;")
        subtitle = QLabel("Local USB camera  ·  ball + clubhead HSV  ·  A.O.A.  ·  carry + roll")
        subtitle.setStyleSheet("color: #86a894; margin-bottom: 6px;")
        root.addWidget(title)
        root.addWidget(subtitle)

        self.speed_card = MetricCard("Ball Speed", "mph")
        self.angle_card = MetricCard("Launch Angle", "degrees")
        self.aoa_card = MetricCard("A.O.A. / Angle of Attack", "deg  ·  Up / Down")
        self.apex_card = MetricCard("Apex Height", "yards")
        self.carry_card = MetricCard("Est. Carry", "yards")
        self.total_card = MetricCard("Total Distance", "yards  ·  carry + roll")
        metrics = QGridLayout()
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(8)
        cards = (
            (self.speed_card, 0, 0),
            (self.angle_card, 0, 1),
            (self.aoa_card, 0, 2),
            (self.apex_card, 1, 0),
            (self.carry_card, 1, 1),
            (self.total_card, 1, 2),
        )
        for card, r, c in cards:
            metrics.addWidget(card, r, c)
        root.addLayout(metrics)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        self.video = VideoPanel()
        left_l.addWidget(self.video, 3)
        self.mask_view = VideoPanel()
        self.mask_view.setMinimumHeight(110)
        self.mask_view.setMaximumHeight(150)
        self.mask_view.show_placeholder("HSV mask preview")
        left_l.addWidget(self.mask_view, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        chart_box = QGroupBox("Predicted Ball Trajectory")
        chart_l = QVBoxLayout(chart_box)
        self.chart = TrajectoryCanvas()
        chart_l.addWidget(self.chart)
        right_l.addWidget(chart_box, 2)
        right_l.addWidget(self._calibration_box(), 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        buttons = QHBoxLayout()
        self.btn_cam = QPushButton("Start Camera")
        self.btn_cam.setObjectName("ghost")
        self.btn_cam.clicked.connect(self._toggle_camera)
        self.btn_sim = QPushButton("Simulate Shot")
        self.btn_sim.clicked.connect(self._simulate_shot)
        self.btn_reset = QPushButton("Reset / Clear")
        self.btn_reset.setObjectName("danger")
        self.btn_reset.clicked.connect(self._reset)
        buttons.addWidget(self.btn_cam)
        buttons.addWidget(self.btn_sim)
        buttons.addWidget(self.btn_reset)
        buttons.addStretch(1)
        self.status = QLabel("Ready. Click Simulate Shot to test indoors without a ball.")
        self.status.setObjectName("status")
        buttons.addWidget(self.status)
        root.addLayout(buttons)

    def _calibration_box(self) -> QGroupBox:
        box = QGroupBox("Calibration Controls")
        outer = QVBoxLayout(box)

        tabs = QTabWidget()
        tabs.addTab(self._hsv_tab("ball"), "Ball HSV")
        tabs.addTab(self._hsv_tab("club"), "Clubhead Marker HSV")
        outer.addWidget(tabs)

        scale_row = QGridLayout()
        scale_row.addWidget(QLabel("Pixel-to-Meter scale"), 0, 0)
        self.scale_box = QDoubleSpinBox()
        self.scale_box.setDecimals(5)
        self.scale_box.setRange(0.00010, 0.25000)
        self.scale_box.setSingleStep(0.00050)
        self.scale_box.setValue(0.00400)
        self.scale_box.setSuffix(" m/px")
        self.scale_box.setToolTip(
            "Meters represented by one image pixel. "
            "Example: a 1.0 m stick spanning 250 px → 0.004 m/px."
        )
        scale_row.addWidget(self.scale_box, 0, 1)
        outer.addLayout(scale_row)

        hint = QLabel(
            "Ball: white → low S-upper, high V-lower.  "
            "Club: put colored tape on the head and match that hue (default is red/orange).  "
            "Mask preview: green = ball, red = club."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7d9a86; font-weight: 400;")
        outer.addWidget(hint)
        return box

    def _hsv_tab(self, kind: str) -> QWidget:
        """kind is 'ball' or 'club' — separate HSV sliders, same layout."""
        page = QWidget()
        grid = QGridLayout(page)
        if kind == "ball":
            defaults = dict(h_low=0, h_high=180, s_low=0, s_high=60, v_low=180, v_high=255)
            store = "sliders"
            labels = "slider_labels"
        else:
            defaults = dict(
                h_low=CLUB_HSV_DEFAULT.h_low,
                h_high=CLUB_HSV_DEFAULT.h_high,
                s_low=CLUB_HSV_DEFAULT.s_low,
                s_high=CLUB_HSV_DEFAULT.s_high,
                v_low=CLUB_HSV_DEFAULT.v_low,
                v_high=CLUB_HSV_DEFAULT.v_high,
            )
            store = "club_sliders"
            labels = "club_slider_labels"

        specs = [
            ("H Lower", "h_low", 0, 180),
            ("H Upper", "h_high", 0, 180),
            ("S Lower", "s_low", 0, 255),
            ("S Upper", "s_high", 0, 255),
            ("V Lower", "v_low", 0, 255),
            ("V Upper", "v_high", 0, 255),
        ]
        setattr(self, store, {})
        setattr(self, labels, {})
        sliders: dict[str, QSlider] = getattr(self, store)
        slider_labels: dict[str, QLabel] = getattr(self, labels)
        for row, (name, key, lo, hi) in enumerate(specs):
            grid.addWidget(QLabel(name), row, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(defaults[key])
            slider.valueChanged.connect(self._sync_hsv)
            sliders[key] = slider
            val = QLabel(str(defaults[key]))
            val.setMinimumWidth(32)
            slider_labels[key] = val
            grid.addWidget(slider, row, 1)
            grid.addWidget(val, row, 2)
        return page

    def _sync_hsv(self) -> None:
        hsv = HSVRange(
            h_low=self.sliders["h_low"].value(),
            s_low=self.sliders["s_low"].value(),
            v_low=self.sliders["v_low"].value(),
            h_high=self.sliders["h_high"].value(),
            s_high=self.sliders["s_high"].value(),
            v_high=self.sliders["v_high"].value(),
        )
        if hsv.h_low > hsv.h_high:
            hsv.h_high = hsv.h_low
            self.sliders["h_high"].blockSignals(True)
            self.sliders["h_high"].setValue(hsv.h_high)
            self.sliders["h_high"].blockSignals(False)
        self.tracker.hsv = hsv
        for key, slider in self.sliders.items():
            self.slider_labels[key].setText(str(slider.value()))

        club = HSVRange(
            h_low=self.club_sliders["h_low"].value(),
            s_low=self.club_sliders["s_low"].value(),
            v_low=self.club_sliders["v_low"].value(),
            h_high=self.club_sliders["h_high"].value(),
            s_high=self.club_sliders["s_high"].value(),
            v_high=self.club_sliders["v_high"].value(),
        )
        if club.h_low > club.h_high:
            club.h_high = club.h_low
            self.club_sliders["h_high"].blockSignals(True)
            self.club_sliders["h_high"].setValue(club.h_high)
            self.club_sliders["h_high"].blockSignals(False)
        self.club_tracker.hsv = club
        for key, slider in self.club_sliders.items():
            self.club_slider_labels[key].setText(str(slider.value()))

    def _try_open_camera(self) -> None:
        if self.camera.open():
            self.status.setText(
                f"Camera ready {self.camera.width}×{self.camera.height} @ {self.camera.fps:.0f} FPS (paused)."
            )
        else:
            self.status.setText("No USB camera found. You can still use Simulate Shot.")

    def _toggle_camera(self) -> None:
        if self.paused:
            if self.camera.cap is None and not self.camera.open():
                self.status.setText("Could not open camera. Use Simulate Shot instead.")
                return
            self.paused = False
            self.btn_cam.setText("Pause Camera")
            self.status.setText("Live tracking. Swing through the frame or use Simulate Shot.")
        else:
            self.paused = True
            self.btn_cam.setText("Start Camera")
            self.video.show_placeholder("Camera paused")
            self.mask_view.show_placeholder("HSV mask preview")
            self.status.setText("Camera paused.")

    def _meters_per_pixel(self) -> float:
        return float(self.scale_box.value())

    def _apply_shot(self, shot: ShotResult, source: str) -> None:
        self.shot_locked = True
        self.speed_card.set_value(f"{shot.ball_speed_mph:.1f}")
        self.angle_card.set_value(f"{shot.launch_angle_deg:.1f}")
        self.aoa_card.set_value(shot.aoa_label)
        self.apex_card.set_value(f"{shot.apex_yards:.1f}")
        self.carry_card.set_value(f"{shot.carry_yards:.1f}")
        self.total_card.set_value(f"{shot.total_yards:.1f}")
        self.chart.set_shot(shot)
        self.status.setText(
            f"{source}: {shot.ball_speed_mph:.1f} mph  ·  "
            f"{shot.launch_angle_deg:.1f}°  ·  {shot.aoa_label}  ·  "
            f"carry {shot.carry_yards:.1f}  ·  total {shot.total_yards:.1f} yd"
        )

    def _simulate_shot(self) -> None:
        self._sync_hsv()
        shot, _centroids, previews = run_mock_pipeline(
            self.tracker,
            self._meters_per_pixel(),
            self.tracker.hsv,
            club_tracker=self.club_tracker,
        )
        self._mock_preview = previews
        self._mock_i = 0
        self._apply_shot(shot, "Mock shot")
        self.live_centroids.clear()
        self.live_stamps.clear()
        self.live_club_centroids.clear()
        self.aoa_locked = shot.aoa_deg

    def _reset(self) -> None:
        self.shot_locked = False
        self.live_centroids.clear()
        self.live_stamps.clear()
        self.live_club_centroids.clear()
        self.aoa_locked = None
        self.miss_frames = 0
        self._mock_preview.clear()
        self.speed_card.set_value("--")
        self.angle_card.set_value("--")
        self.aoa_card.set_value("--")
        self.apex_card.set_value("--")
        self.carry_card.set_value("--")
        self.total_card.set_value("--")
        self.chart.set_shot(None)
        self.status.setText("Cleared. Ready for the next shot.")
        if self.paused:
            self.video.show_placeholder("Camera paused")
            self.mask_view.show_placeholder("HSV mask preview")

    def _finalize_live_shot(self) -> None:
        if len(self.live_centroids) < 3:
            self.live_centroids.clear()
            self.live_stamps.clear()
            self.live_club_centroids.clear()
            return
        speed_mps, launch_deg = kinematics_from_centroids(
            self.live_centroids, self.live_stamps, self._meters_per_pixel()
        )
        if self.aoa_locked is None:
            self.aoa_locked = angle_of_attack_deg(self.live_club_centroids)
        self.live_centroids.clear()
        self.live_stamps.clear()
        self.live_club_centroids.clear()
        if speed_mps < 1.5:
            return
        shot = simulate_flight(speed_mps, launch_deg, aoa_deg=self.aoa_locked)
        self._apply_shot(shot, "Tracked shot")

    def _on_tick(self) -> None:
        if self._mock_preview and self._mock_i < len(self._mock_preview):
            frame = self._mock_preview[self._mock_i]
            self._mock_i += 1
            self.video.show_bgr(frame)
            ball_det = self.tracker.detect(frame)
            club_det = self.club_tracker.detect(frame)
            self.mask_view.show_bgr(combine_masks(ball_det.mask, club_det.mask))
            if self._mock_i >= len(self._mock_preview):
                self._mock_preview.clear()
            return

        if self.paused:
            return

        frame = self.camera.read()
        if frame is None:
            self.video.show_placeholder("Camera frame unavailable")
            return

        ball_det = self.tracker.detect(frame)
        club_det = self.club_tracker.detect(frame)
        self.video.show_bgr(annotate_ball_and_club(frame, ball_det, club_det))
        mask_bgr = combine_masks(ball_det.mask, club_det.mask)
        if ball_det.found:
            cv2.circle(mask_bgr, (int(ball_det.cx), int(ball_det.cy)), 6, (0, 255, 0), -1)
        if club_det.found:
            cv2.circle(mask_bgr, (int(club_det.cx), int(club_det.cy)), 6, (0, 0, 255), -1)
        self.mask_view.show_bgr(mask_bgr)

        if self.shot_locked:
            return

        now = time.perf_counter()
        # Club path is collected *before* the ball is seen (entry into impact).
        if club_det.found and not self.live_centroids:
            self.live_club_centroids.append((club_det.cx, club_det.cy))
            if len(self.live_club_centroids) > 40:
                self.live_club_centroids = self.live_club_centroids[-40:]

        if ball_det.found:
            if self.aoa_locked is None and len(self.live_club_centroids) >= 2:
                self.aoa_locked = angle_of_attack_deg(self.live_club_centroids)
            self.miss_frames = 0
            self.live_centroids.append((ball_det.cx, ball_det.cy))
            self.live_stamps.append(now)
            if len(self.live_centroids) > 40:
                self.live_centroids = self.live_centroids[-40:]
                self.live_stamps = self.live_stamps[-40:]
        else:
            self.miss_frames += 1
            if self.miss_frames >= 8 and self.live_centroids:
                self._finalize_live_shot()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        self.camera.release()
        super().closeEvent(event)
