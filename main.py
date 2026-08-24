"""Launch the DIY All-in-One Golf Simulator (A.O.A. + carry/roll dashboard)."""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from golf_sim.ui import GolfSimulatorWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("DIY Golf Simulator")
    window = GolfSimulatorWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
