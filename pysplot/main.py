def main():
    import sys
    import serial
    import serial.tools.list_ports
    import numpy as np
    from PyQt6 import QtWidgets, QtCore
    import pyqtgraph as pg
    import datetime
    import argparse
    import threading

    DARK_BG = "#1e1e1e"
    DARK_TEXT = "#e0e0e0"
    BUFF_SIZE = 1000
    UPDATE_INTERVAL = 1  # ms
    SERIAL_TIMEOUT = 0.1
    AUTOSCALE_SAMPLES = 10  # Use samples 6-15
    SKIP_INITIAL_SAMPLES = 5  # Skip first 5 samples

    pg.setConfigOption("background", DARK_BG)
    pg.setConfigOption("foreground", DARK_TEXT)

    # ===== CLI ARGUMENT PARSING =====
    def parse_args():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Real-time serial data plotter with dark theme"
        )
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

    # ===== SERIAL PORT SETUP =====
    def get_port(port_arg):
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

    # ===== CHECK FOR PIPED INPUT =====
    def has_piped_input():
        """Check if stdin is piped (not a TTY)."""
        return not sys.stdin.isatty()

    args = parse_args()
    baudrate = args.baudrate
    buff_size = args.size
    use_pipe = has_piped_input()

    if use_pipe:
        print("Stdin mode: Reading from piped input")
    else:
        port = get_port(args.port)
        print(f"Serial mode: Port: {port} | Baudrate: {baudrate} | Buffer: {buff_size}")

    # ===== SERIAL WORKER THREAD =====
    class SerialWorker(QtCore.QObject):
        """Worker thread for non-blocking serial communication."""

        data_received = QtCore.pyqtSignal(np.ndarray)
        error_occurred = QtCore.pyqtSignal(str)

        def __init__(self, port, baudrate, timeout=SERIAL_TIMEOUT):
            super().__init__()
            try:
                self.ser = serial.Serial(port, baudrate, timeout=timeout)
                self.ser.reset_input_buffer()
            except serial.SerialException as e:
                self.error_occurred.emit(f"Serial error: {e}")
                raise
            self.queue = []
            self.running = True
            self.delimiter = None

        @staticmethod
        def detect_delimiter(line):
            """Auto-detect delimiter (tab, comma, or space)."""
            for delim in ["\t", ",", " ", ";"]:
                if delim in line:
                    return delim
            return "\t"  # default

        def detect_signals(self):
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
            print("Warning: Could not auto-detect signals, assuming 1 with tab delimiter")
            self.delimiter = "\t"
            return 1

        def run(self):
            """Main worker loop - continuously read serial data into queue."""
            while self.running:
                try:
                    # Read all available data
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

                    # Emit queued data
                    while self.queue:
                        self.data_received.emit(self.queue.pop(0))

                    QtCore.QThread.msleep(UPDATE_INTERVAL)
                except UnicodeDecodeError:
                    pass
                except Exception as e:
                    self.error_occurred.emit(f"Worker error: {e}")

        def stop(self):
            """Stop worker and close serial port."""
            self.running = False
            if self.ser and self.ser.is_open:
                self.ser.close()

    # ===== STDIN WORKER THREAD =====
    class StdinWorker(QtCore.QObject):
        """Worker thread for reading from piped stdin."""

        data_received = QtCore.pyqtSignal(np.ndarray)
        error_occurred = QtCore.pyqtSignal(str)

        def __init__(self):
            super().__init__()
            import fcntl
            import os

            self.queue = []
            self.running = True
            self.delimiter = None
            self.fd = sys.stdin.fileno()
            self.buffer = ""  # Shared buffer to preserve data between detect and run

            # Make stdin non-blocking
            try:
                flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
                fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            except Exception as e:
                print(f"Warning: Could not set non-blocking mode: {e}")

        @staticmethod
        def detect_delimiter(line):
            """Auto-detect delimiter (tab, comma, or space)."""
            for delim in ["\t", ",", " ", ";"]:
                if delim in line:
                    return delim
            return "\t"  # default

        def detect_signals(self):
            """Blocking wait for first line to determine signal count."""
            print("Waiting for data on stdin...")
            import time
            import os

            while True:
                try:
                    # Read chunks until we get a newline
                    chunk = os.read(self.fd, 4096)
                    if chunk:
                        self.buffer += chunk.decode("utf-8", errors="ignore")

                        if "\n" in self.buffer:
                            # Grab the first line
                            line, remainder = self.buffer.split("\n", 1)
                            line = line.strip()

                            # Put remainder back in buffer for run()
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

                                    # Queue this first sample so it's not lost
                                    self.queue.append(values)
                                    return count
                                except ValueError:
                                    pass  # Header or garbage? continue
                    else:
                        # EOF or no data yet; wait briefly
                        time.sleep(0.01)

                except BlockingIOError:
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Error detecting signals: {e}")
                    return 1  # Fallback if something goes wrong

        def run(self):
            """Main worker loop - continuously read stdin data."""
            import os
            import time

            while self.running:
                try:
                    # 1. Read all available data into self.buffer (Drain the pipe)
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

                    # 2. Process complete lines in buffer
                    while "\n" in self.buffer:
                        line, self.buffer = self.buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                values = np.array(
                                    [float(v.strip()) for v in line.split(self.delimiter or "\t")]
                                )
                                self.queue.append(values)
                            except (ValueError, IndexError):
                                pass

                    # 3. Emit queued data
                    while self.queue:
                        self.data_received.emit(self.queue.pop(0))

                    # 4. Sleep briefly
                    time.sleep(0.001)

                except Exception as e:
                    self.error_occurred.emit(f"Worker error: {e}")
                    time.sleep(0.1)

        def stop(self):
            self.running = False

    # ===== STACKED PLOTS WINDOW CLASS =====
    class StackedPlotsWindow(QtWidgets.QMainWindow):
        """Single window with stacked plots sharing time axis."""

        def __init__(self, num_signals, buff_size):
            super().__init__()
            self.num_signals = num_signals
            self.buff_size = buff_size
            self.setWindowTitle("pysplot - Serial Data Plotter")
            self.resize(1000, 100 + 150 * num_signals)
            self.frozen = False

            # Autoscale tracking
            self.sample_count = 0
            self.autoscale_applied = False
            self.autoscale_samples = []
            self.autoscale_enabled = True

            # Create central widget with main layout
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            main_layout = QtWidgets.QVBoxLayout(central)
            main_layout.setContentsMargins(0, 0, 0, 0)

            # Create button widget
            btn_widget = QtWidgets.QWidget()
            btn_layout = QtWidgets.QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(10, 5, 10, 5)

            # Create freeze button
            self.freeze_btn = QtWidgets.QPushButton("Freeze")
            self.freeze_btn.setCheckable(True)
            self.freeze_btn.setMaximumWidth(100)
            self.freeze_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #404040;
                    color: {DARK_TEXT};
                    border: 1px solid #606060;
                    padding: 5px;
                    border-radius: 3px;
                }}
                QPushButton:checked {{
                    background-color: #ff6b6b;
                    color: #000000;
                }}
            """)
            self.freeze_btn.toggled.connect(self.on_freeze_toggled)
            btn_layout.addWidget(self.freeze_btn)

            # Create rescale button
            self.rescale_btn = QtWidgets.QPushButton("Rescale")
            self.rescale_btn.setMaximumWidth(100)
            self.rescale_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #2d5a7a;
                    color: {DARK_TEXT};
                    border: 1px solid #4a7c9a;
                    padding: 5px;
                    border-radius: 3px;
                }}
                QPushButton:hover {{
                    background-color: #3d7a9a;
                }}
            """)
            self.rescale_btn.clicked.connect(self.on_rescale)
            btn_layout.addWidget(self.rescale_btn)

            # Create export button (hidden by default)
            self.export_btn = QtWidgets.QPushButton("Export CSV")
            self.export_btn.setMaximumWidth(120)
            self.export_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #2d5a2d;
                    color: {DARK_TEXT};
                    border: 1px solid #4a7c4a;
                    padding: 5px;
                    border-radius: 3px;
                }}
                QPushButton:hover {{
                    background-color: #3d7a3d;
                }}
            """)
            self.export_btn.clicked.connect(self.on_export_csv)
            self.export_btn.setVisible(False)
            btn_layout.addWidget(self.export_btn)

            btn_layout.addStretch()
            btn_widget.setStyleSheet(f"QWidget {{ background-color: {DARK_BG}; }}")
            main_layout.addWidget(btn_widget)

            # Create graphics layout widget for plots
            layout_widget = pg.GraphicsLayoutWidget()
            main_layout.addWidget(layout_widget)

            # Create data buffers
            self.x = np.arange(buff_size)
            self.sample_buffers = np.zeros((num_signals, buff_size))

            # Store plots and curves
            self.plots = []
            self.curves = []

            # Create stacked plots
            self.bottom_plot = None
            for i in range(num_signals):
                plot = layout_widget.addPlot(row=i, col=0, title=f"Signal {i + 1}")
                plot.getViewBox().setBackgroundColor(DARK_BG)
                plot.showGrid(x=True, y=True, alpha=0.2)

                # Style axes
                plot.getAxis("left").setPen(pg.mkPen(color=DARK_TEXT, width=1))
                plot.getAxis("left").setTextPen(pg.mkPen(color=DARK_TEXT))
                plot.getAxis("bottom").setPen(pg.mkPen(color=DARK_TEXT, width=1))
                plot.getAxis("bottom").setTextPen(pg.mkPen(color=DARK_TEXT))

                # Only show x-axis label on bottom plot
                if i == num_signals - 1:
                    plot.setLabel("bottom", "Sample", color=DARK_TEXT)
                    self.bottom_plot = plot
                else:
                    plot.getAxis("bottom").hide()

                plot.setLabel("left", f"Signal {i + 1}", color=DARK_TEXT)

                # Create curve
                color = pg.intColor(i, hues=num_signals)
                curve = plot.plot(pen=color)

                self.plots.append(plot)
                self.curves.append(curve)

            # Link x-axes together
            for i in range(1, num_signals):
                self.plots[i].setXLink(self.plots[0])

            # Apply dark theme to window
            central.setStyleSheet(f"QWidget {{ background-color: {DARK_BG}; color: {DARK_TEXT}; }}")
            self.setStyleSheet(
                f"QMainWindow {{ background-color: {DARK_BG}; color: {DARK_TEXT}; }}"
            )

        def on_freeze_toggled(self, checked):
            """Handle freeze button toggle."""
            self.frozen = checked
            self.export_btn.setVisible(checked)
            if checked:
                self.freeze_btn.setText("Frozen")
            else:
                self.freeze_btn.setText("Freeze")

        def apply_autoscale(self):
            """Apply autoscaling based on current buffer data."""
            if not self.autoscale_enabled:
                return

            for i, plot in enumerate(self.plots):
                y_data = self.sample_buffers[i]
                # Only consider non-zero values
                valid_data = y_data[y_data != 0] if np.any(y_data != 0) else y_data

                if len(valid_data) > 0:
                    y_min, y_max = np.min(valid_data), np.max(valid_data)
                    y_range = y_max - y_min

                    # Add 10% padding
                    padding = y_range * 0.1 if y_range > 0 else 0.5
                    plot.setYRange(y_min - padding, y_max + padding, padding=0)

        def rescale_to_first_10_percent(self):
            """Rescale plots based on first 10% of samples in buffer."""
            num_samples_to_use = max(1, self.buff_size // 10)

            for i, plot in enumerate(self.plots):
                # Get first 10% of buffer (most recent data in circular buffer)
                y_data = self.sample_buffers[i, :num_samples_to_use]

                # Only consider non-zero values
                valid_data = y_data[y_data != 0] if np.any(y_data != 0) else y_data

                if len(valid_data) > 0:
                    y_min, y_max = np.min(valid_data), np.max(valid_data)
                    y_range = y_max - y_min

                    # Add 10% padding
                    padding = y_range * 0.1 if y_range > 0 else 0.5
                    plot.setYRange(y_min - padding, y_max + padding, padding=0)

            print(
                f"Rescaled to first {num_samples_to_use} samples ({num_samples_to_use / self.buff_size * 100:.1f}% of buffer)"
            )

        def on_rescale(self):
            """Handle rescale button click."""
            self.rescale_to_first_10_percent()

        def update_data(self, values):
            """Update all plots with new data."""
            if not self.frozen and len(values) == self.num_signals:
                # Track samples for initial autoscaling trigger
                if not self.autoscale_applied:
                    self.sample_count += 1

                    # Skip first SKIP_INITIAL_SAMPLES, collect next AUTOSCALE_SAMPLES
                    if self.sample_count > SKIP_INITIAL_SAMPLES:
                        self.autoscale_samples.append(values.copy())

                    # Enable continuous autoscaling once we have enough samples
                    if len(self.autoscale_samples) == AUTOSCALE_SAMPLES:
                        print(
                            f"Autoscale enabled based on samples {SKIP_INITIAL_SAMPLES + 1}-{SKIP_INITIAL_SAMPLES + AUTOSCALE_SAMPLES}"
                        )
                        self.autoscale_applied = True

                # Update buffers
                self.sample_buffers = np.column_stack([values, self.sample_buffers[:, :-1]])

                # Update curves
                for i, curve in enumerate(self.curves):
                    curve.setData(self.x, self.sample_buffers[i])

                # Apply continuous autoscaling
                if self.autoscale_enabled and self.autoscale_applied:
                    self.apply_autoscale()

        def on_export_csv(self):
            """Export current frozen data to CSV."""
            filename, ok = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export Frozen Data",
                f"splotdata_frozen_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.csv",
                "CSV Files (*.csv)",
            )
            if ok and filename:
                try:
                    np.savetxt(
                        filename,
                        self.sample_buffers.T,
                        delimiter=",",
                        header=",".join(f"Signal {i + 1}" for i in range(self.num_signals)),
                        comments="",
                    )
                    print(f"Data exported: {filename}")
                except Exception as e:
                    print(f"Error exporting data: {e}")
                    QtWidgets.QMessageBox.critical(
                        self, "Export Error", f"Failed to export data: {e}"
                    )

    # ===== GUI SETUP =====
    app = QtWidgets.QApplication([])

    # ===== INITIALIZE WORKER THREAD =====
    try:
        if use_pipe:
            worker = StdinWorker()
            # Wait for the first line of data to determine count
            num_signals = worker.detect_signals()
        else:
            port = get_port(args.port)
            worker = SerialWorker(port, baudrate)
            num_signals = worker.detect_signals()
    except (serial.SerialException, Exception) as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

    # Create stacked plots window (immediately, even in pipe mode)
    window = StackedPlotsWindow(num_signals, buff_size)
    window.show()

    # ===== WORKER THREAD SETUP =====
    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    # ===== SIGNAL HANDLING =====
    def on_data_received(values):
        """Update plots with new data."""
        window.update_data(values)

    def on_error(msg):
        """Handle worker errors."""
        print(f"Error: {msg}")

    def on_signals_detected(num_signals_actual):
        """Handle detection of actual signal count in piped mode."""
        if use_pipe and num_signals_actual != window.num_signals:
            print(f"Resizing window for {num_signals_actual} signals")
            window.num_signals = num_signals_actual
            window.resize(1000, 100 + 150 * num_signals_actual)
            # Recreate buffers and plots if needed
            window.sample_buffers = np.zeros((num_signals_actual, buff_size))

    worker.data_received.connect(on_data_received)
    worker.error_occurred.connect(on_error)
    if hasattr(worker, "signals_detected"):
        worker.signals_detected.connect(on_signals_detected)

    # ===== CLEANUP ON EXIT =====
    def on_close():
        """Save data and clean up resources."""
        worker.stop()
        thread.quit()
        thread.wait()

    app.aboutToQuit.connect(on_close)

    thread.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
