# DIY All-in-One Golf Simulator

Local desktop app for a USB-camera golf simulator: live HSV ball tracking, launch kinematics, and 2D projectile motion with quadratic drag.

## Install

Use Python 3.10 or newer.

```bash
cd Golf
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Run

```bash
python main.py
```

### Indoor mock test (no ball, camera optional)

1. Start the app.
2. Optionally set **Pixel-to-Meter** (default `0.004` m/px is fine for the mock).
3. Click **Simulate Shot**.
4. The video panel plays a synthetic launch, the tracker finds the ball, physics fills **Ball Speed**, **Launch Angle**, **Apex Height**, and **Estimated Carry**, and the canvas draws the flight arc.

**Reset / Clear** wipes metrics and the trajectory so you can run another mock or a live swing.

### Live USB camera

1. Click **Start Camera**.
2. Aim the camera at the ball flight (side-on works best for launch angle).
3. Move the **HSV** sliders until the mask preview shows only the ball (white balls: low S-upper, high V-lower).
4. Calibrate **Pixel-to-Meter**: measure a known length in the plane of the ball and divide meters by pixels.
5. Hit. After the ball leaves the frame, speed/angle are computed from centroid motion and the carry curve is plotted.

Pause the camera anytime; the feed shows a placeholder until you start it again.

## Layout

| Path | Role |
| --- | --- |
| `main.py` | App entry |
| `golf_sim/ui.py` | PyQt5 dashboard |
| `golf_sim/tracker.py` | HSV mask, blur, `findContours` |
| `golf_sim/physics.py` | Speed/angle + drag trajectory |
| `golf_sim/mock_shot.py` | Indoor synthetic shot pipeline |
| `golf_sim/camera.py` | USB capture (tries 1080p / 60 FPS) |

## Notes

- This is a DIY estimator, not a launch-monitor replacement. Side-on calibration and lighting dominate accuracy.
- Air model: \(F_d = \tfrac{1}{2} C_d \rho A |v|^2\) with a typical golf-ball \(C_d \approx 0.25\) (no Magnus lift).
