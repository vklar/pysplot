def main():
    import sys
    import serial
    import serial.tools.list_ports
    import numpy as np
    from PyQt6 import QtWidgets, QtCore
    import pyqtgraph as pg
    import datetime
    import argparse

    DARK_BG = "#1e1e1e"
    DARK_TEXT = "#e0e0e0"
    BUFF_SIZE = 1000
    UPDATE_INTERVAL = 1  # ms
    SERIAL_TIMEOUT = 0.1

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

    args = parse_args()
    port = get_port(args.port)
    baudrate = args.baudrate
    buff_size = args.size

    print(f"Port: {port} | Baudrate: {baudrate} | Buffer: {buff_size}")

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
                plot = layout_widget.addPlot(
                    row=i, col=0,
                    title=f"Signal {i + 1}"
                )
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
            self.setStyleSheet(f"QMainWindow {{ background-color: {DARK_BG}; color: {DARK_TEXT}; }}")

        def on_freeze_toggled(self, checked):
            """Handle freeze button toggle."""
            self.frozen = checked
            self.export_btn.setVisible(checked)
            if checked:
                self.freeze_btn.setText("Frozen")
            else:
                self.freeze_btn.setText("Freeze")

        def update_data(self, values):
            """Update all plots with new data."""
            if not self.frozen and len(values) == self.num_signals:
                # Update buffers
                self.sample_buffers = np.column_stack([values, self.sample_buffers[:, :-1]])
                
                # Update curves
                for i, curve in enumerate(self.curves):
                    curve.setData(self.x, self.sample_buffers[i])

        def on_export_csv(self):
            """Export current frozen data to CSV."""
            filename, ok = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export Frozen Data",
                f"splotdata_frozen_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.csv",
                "CSV Files (*.csv)"
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
                    QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export data: {e}")

    # ===== INITIALIZE WORKER THREAD =====
    try:
        worker = SerialWorker(port, baudrate)
        num_signals = worker.detect_signals()
    except serial.SerialException as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

    # ===== GUI SETUP =====
    app = QtWidgets.QApplication([])

    # Create stacked plots window
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

    worker.data_received.connect(on_data_received)
    worker.error_occurred.connect(on_error)

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
