import sys
import time
import os
import fcntl
import numpy as np
import serial
import serial.tools.list_ports
from PyQt6 import QtCore

from pysplot.config import (
    UPDATE_INTERVAL,
    SERIAL_TIMEOUT,
    DELIMITER_OPTIONS,
    DEFAULT_DELIMITER,
)


class SerialWorker(QtCore.QObject):
    """Worker thread for non-blocking serial communication."""

    data_received = QtCore.pyqtSignal(np.ndarray)  # type: ignore[assignment]
    error_occurred = QtCore.pyqtSignal(str)  # type: ignore[assignment]

    def __init__(self, port: str, baudrate: int, timeout: float = SERIAL_TIMEOUT) -> None:
        super().__init__()
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            self.ser.reset_input_buffer()
        except serial.SerialException as e:
            self.error_occurred.emit(f"Serial error: {e}")
            raise
        self.queue: list[np.ndarray] = []
        self.running = True
        self.delimiter: str | None = None

    @staticmethod
    def detect_delimiter(line: str) -> str:
        """Auto-detect delimiter (tab, comma, or space)."""
        for delim in DELIMITER_OPTIONS:
            if delim in line:
                return delim
        return DEFAULT_DELIMITER

    def detect_signals(self) -> int:
        """Read first line to determine number of signals and delimiter."""
        max_attempts = 100
        attempts = 0
        while self.running and attempts < max_attempts:
            try:
                line = self.ser.readline().decode().strip()
                if line:
                    self.delimiter = self.detect_delimiter(line)
                    num = len(line.split(self.delimiter))
                    print(f"Detected {num} signal(s) with delimiter: {repr(self.delimiter)}")
                    return num
            except (UnicodeDecodeError, ValueError):
                pass
            attempts += 1
        print(
            f"Warning: Could not auto-detect signals, assuming 1 with {repr(DEFAULT_DELIMITER)} delimiter"
        )
        self.delimiter = DEFAULT_DELIMITER
        return 1

    def run(self) -> None:
        """Main worker loop - continuously read serial data into queue."""
        while self.running:
            try:
                while self.ser.in_waiting:
                    line = self.ser.readline().decode().strip()
                    if line:
                        try:
                            values = np.array(
                                [float(v.strip()) for v in line.split(self.delimiter)]
                            )
                            self.queue.append(values)
                        except (ValueError, IndexError):
                            pass

                while self.queue:
                    self.data_received.emit(self.queue.pop(0))

                QtCore.QThread.msleep(UPDATE_INTERVAL)
            except UnicodeDecodeError:
                pass
            except Exception as e:
                self.error_occurred.emit(f"Worker error: {e}")

    def stop(self) -> None:
        """Stop worker and close serial port."""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


class StdinWorker(QtCore.QObject):
    """Worker thread for reading from piped stdin."""

    data_received = QtCore.pyqtSignal(np.ndarray)  # type: ignore[assignment]
    error_occurred = QtCore.pyqtSignal(str)  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.queue: list[np.ndarray] = []
        self.running = True
        self.delimiter: str | None = None
        self.fd = sys.stdin.fileno()
        self.buffer = ""

        try:
            flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except Exception as e:
            print(f"Warning: Could not set non-blocking mode: {e}")

    @staticmethod
    def detect_delimiter(line: str) -> str:
        """Auto-detect delimiter (tab, comma, or space)."""
        for delim in DELIMITER_OPTIONS:
            if delim in line:
                return delim
        return DEFAULT_DELIMITER

    def detect_signals(self) -> int:
        """Blocking wait for first line to determine signal count."""
        print("Waiting for data on stdin...")

        while True:
            try:
                chunk = os.read(self.fd, 4096)
                if chunk:
                    self.buffer += chunk.decode("utf-8", errors="ignore")

                    if "\n" in self.buffer:
                        line, remainder = self.buffer.split("\n", 1)
                        line = line.strip()
                        self.buffer = remainder

                        if line:
                            self.delimiter = self.detect_delimiter(line)
                            try:
                                values = np.array(
                                    [float(v.strip()) for v in line.split(self.delimiter)]
                                )
                                count = len(values)
                                print(
                                    f"Detected {count} signals with delimiter: {repr(self.delimiter)}"
                                )
                                self.queue.append(values)
                                return count
                            except ValueError:
                                pass
                else:
                    time.sleep(0.01)

            except BlockingIOError:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error detecting signals: {e}")
                return 1

    def run(self) -> None:
        """Main worker loop - continuously read stdin data."""
        while self.running:
            try:
                raw_data = b""
                while True:
                    try:
                        chunk = os.read(self.fd, 8192)
                        if not chunk:
                            break
                        raw_data += chunk
                    except BlockingIOError:
                        break
                    except Exception:
                        break

                if raw_data:
                    self.buffer += raw_data.decode("utf-8", errors="ignore")

                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            values = np.array(
                                [
                                    float(v.strip())
                                    for v in line.split(self.delimiter or DEFAULT_DELIMITER)
                                ]
                            )
                            self.queue.append(values)
                        except (ValueError, IndexError):
                            pass

                while self.queue:
                    self.data_received.emit(self.queue.pop(0))

                time.sleep(0.001)

            except Exception as e:
                self.error_occurred.emit(f"Worker error: {e}")
                time.sleep(0.1)

    def stop(self) -> None:
        self.running = False
