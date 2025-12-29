import argparse
import sys
from typing import Optional

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt6 import QtWidgets, QtCore

from pysplot.config import (
    DARK_BG,
    DARK_TEXT,
    BUFF_SIZE,
)
from pysplot.workers import SerialWorker, StdinWorker
from pysplot.ui import StackedPlotsWindow


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-time serial data plotter with dark theme")
    parser.add_argument(
        "port",
        nargs="?",
        help="Serial port (auto-detect if omitted)",
    )
    parser.add_argument(
        "-b",
        "--baudrate",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=BUFF_SIZE,
        help=f"Buffer size (default: {BUFF_SIZE})",
    )
    return parser.parse_args()


def get_port(port_arg: Optional[str]) -> str:
    """Get serial port, auto-detect if not provided."""
    if port_arg:
        return port_arg

    usb_ports = [
        p.device
        for p in serial.tools.list_ports.comports()
        if "USB" in p.device or "ACM" in p.device
    ]
    if not usb_ports:
        print("Error: No USB ports found")
        sys.exit(1)
    return usb_ports[0]


def has_piped_input() -> bool:
    """Check if stdin is piped (not a TTY)."""
    return not sys.stdin.isatty()


def main() -> None:
    """Main entry point for pysplot."""
    app = QtWidgets.QApplication([])
    pg.setConfigOption("background", DARK_BG)
    pg.setConfigOption("foreground", DARK_TEXT)

    args = parse_args()
    baudrate = args.baudrate
    buff_size = args.size
    use_pipe = has_piped_input()

    if use_pipe:
        print("Stdin mode: Reading from piped input")
    else:
        port = get_port(args.port)
        print(f"Serial mode: Port: {port} | Baudrate: {baudrate} | Buffer: {buff_size}")

    try:
        if use_pipe:
            worker = StdinWorker()
            num_signals = worker.detect_signals()
        else:
            port = get_port(args.port)
            worker = SerialWorker(port, baudrate)
            num_signals = worker.detect_signals()
    except (serial.SerialException, Exception) as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

    window = StackedPlotsWindow(num_signals, buff_size)
    window.show()

    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    def on_data_received(values: np.ndarray) -> None:
        """Update plots with new data."""
        window.update_data(values)

    def on_error(msg: str) -> None:
        """Handle worker errors."""
        print(f"Error: {msg}")

    def on_signals_detected(num_signals_actual: int) -> None:
        """Handle detection of actual signal count in piped mode."""
        if use_pipe and num_signals_actual != window.num_signals:
            print(f"Resizing window for {num_signals_actual} signals")
            window.num_signals = num_signals_actual
            window.resize(1000, 100 + 150 * num_signals_actual)
            window.sample_buffers = np.zeros((num_signals_actual, buff_size))

    worker.data_received.connect(on_data_received)
    worker.error_occurred.connect(on_error)
    if hasattr(worker, "signals_detected"):
        worker.signals_detected.connect(on_signals_detected)

    def on_close() -> None:
        """Save data and clean up resources."""
        worker.stop()
        thread.quit()
        thread.wait()

    app.aboutToQuit.connect(on_close)

    thread.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
