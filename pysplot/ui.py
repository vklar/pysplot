import datetime
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui

from pysplot.config import (
    DARK_BG,
    DARK_TEXT,
    BUFF_SIZE,
    AUTOSCALE_SAMPLES,
    SKIP_INITIAL_SAMPLES,
    DEFAULT_DELIMITER,
)


class StackedPlotsWindow(QtWidgets.QMainWindow):
    """Single window with stacked plots sharing time axis."""

    def __init__(self, num_signals: int, buff_size: int = BUFF_SIZE) -> None:
        super().__init__()
        self.num_signals = num_signals
        self.buff_size = buff_size
        self.setWindowTitle("pysplot - Serial Data Plotter")
        self.resize(1000, 100 + 150 * num_signals)
        self.frozen = False

        self.sample_count = 0
        self.autoscale_applied = False
        self.autoscale_samples: list[np.ndarray] = []
        self.autoscale_enabled = True

        self.plot_visible = [True] * num_signals
        self.plot_containers: list[
            tuple[QtWidgets.QWidget, pg.PlotWidget, QtWidgets.QPushButton, QtWidgets.QWidget]
        ] = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        btn_widget = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(10, 5, 10, 5)

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

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)

        self.x_axis = np.arange(buff_size)
        self.sample_buffers = np.zeros((num_signals, buff_size))

        self.plots: list[pg.PlotWidget] = []
        self.curves: list[pg.PlotDataItem] = []

        for i in range(num_signals):
            plot_container = QtWidgets.QWidget()
            plot_layout = QtWidgets.QVBoxLayout(plot_container)
            plot_layout.setContentsMargins(0, 0, 0, 0)
            plot_layout.setSpacing(0)

            header = QtWidgets.QWidget()
            header_layout = QtWidgets.QHBoxLayout(header)
            header_layout.setContentsMargins(10, 5, 10, 5)

            min_btn = QtWidgets.QPushButton("−")
            min_btn.setMaximumWidth(30)
            min_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #404040;
                    color: {DARK_TEXT};
                    border: 1px solid #606060;
                    padding: 2px;
                    border-radius: 2px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #505050;
                }}
            """)

            title_label = QtWidgets.QLabel(f"Signal {i + 1}")
            title_label.setStyleSheet(f"color: {DARK_TEXT}; font-weight: bold;")

            header_layout.addWidget(min_btn)
            header_layout.addWidget(title_label)
            header_layout.addStretch()

            header.setStyleSheet(f"QWidget {{ background-color: #2a2a2a; }}")

            plot_layout.addWidget(header)

            plot_widget = pg.PlotWidget()
            plot_widget.getViewBox().setBackgroundColor(DARK_BG)
            plot_widget.showGrid(x=True, y=True, alpha=0.2)

            plot_widget.getAxis("left").setPen(pg.mkPen(color=DARK_TEXT, width=1))
            plot_widget.getAxis("left").setTextPen(pg.mkPen(color=DARK_TEXT))
            plot_widget.getAxis("bottom").setPen(pg.mkPen(color=DARK_TEXT, width=1))
            plot_widget.getAxis("bottom").setTextPen(pg.mkPen(color=DARK_TEXT))

            if i == num_signals - 1:
                plot_widget.setLabel("bottom", "Sample", color=DARK_TEXT)
            else:
                plot_widget.getAxis("bottom").hide()

            plot_widget.setLabel("left", f"Signal {i + 1}", color=DARK_TEXT)
            plot_widget.setMaximumHeight(500)
            plot_widget.setMinimumHeight(80)

            color = pg.intColor(i, hues=num_signals)
            curve = plot_widget.plot(pen=color)

            plot_layout.addWidget(plot_widget, 1)

            resize_handle = QtWidgets.QWidget()
            resize_handle.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)
            resize_handle.setFixedHeight(8)
            resize_handle.setStyleSheet(f"QWidget {{ background-color: #3a3a3a; }}")
            plot_layout.addWidget(resize_handle)

            plot_index = i
            min_btn.clicked.connect(
                lambda checked, idx=plot_index: self.toggle_plot_visibility(idx)
            )

            self.plots.append(plot_widget)
            self.curves.append(curve)
            self.plot_containers.append((plot_container, plot_widget, min_btn, resize_handle))

            scroll_layout.addWidget(plot_container)

            self.setup_resize_handler(plot_container, plot_widget, resize_handle, plot_index)

        for i in range(1, num_signals):
            self.plots[i].setXLink(self.plots[0])

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        scroll_area.setStyleSheet(f"QWidget {{ background-color: {DARK_BG}; color: {DARK_TEXT}; }}")
        scroll_widget.setStyleSheet(f"QWidget {{ background-color: {DARK_BG}; }}")
        central.setStyleSheet(f"QWidget {{ background-color: {DARK_BG}; color: {DARK_TEXT}; }}")
        self.setStyleSheet(f"QMainWindow {{ background-color: {DARK_BG}; color: {DARK_TEXT}; }}")

    def setup_resize_handler(
        self,
        container: QtWidgets.QWidget,
        plot_widget: pg.PlotWidget,
        handle: QtWidgets.QWidget,
        index: int,
    ) -> None:
        """Setup drag-to-resize functionality for a plot."""
        if not hasattr(self, "resize_data"):
            self.resize_data: dict[int, dict[str, float]] = {}

        class ResizeHandle(QtWidgets.QWidget):
            def __init__(self, parent_window: "StackedPlotsWindow", pw: pg.PlotWidget, idx: int):
                super().__init__()
                self.parent_window = parent_window
                self.plot_widget = pw
                self.index = idx
                self.dragging = False

            def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
                self.dragging = True
                self.parent_window.resize_data[self.index] = {
                    "start_y": event.globalPosition().y(),
                    "start_height": self.plot_widget.height(),
                }

            def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
                if self.dragging and self.index in self.parent_window.resize_data:
                    delta = (
                        event.globalPosition().y()
                        - self.parent_window.resize_data[self.index]["start_y"]
                    )
                    new_height = max(
                        80,
                        self.parent_window.resize_data[self.index]["start_height"] + delta,
                    )
                    self.plot_widget.setFixedHeight(int(new_height))

            def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
                self.dragging = False
                if self.index in self.parent_window.resize_data:
                    del self.parent_window.resize_data[self.index]

        parent = handle.parent()
        if parent is None:
            return
        parent_layout = parent.layout()  # type: ignore[assignment]
        if parent_layout is None:
            return
        idx = parent_layout.indexOf(handle)
        handle.deleteLater()

        new_handle = ResizeHandle(self, plot_widget, index)
        new_handle.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)
        new_handle.setFixedHeight(8)
        new_handle.setStyleSheet(f"QWidget {{ background-color: #3a3a3a; }}")
        parent_layout.insertWidget(idx, new_handle)

        container_ref = self.plot_containers[index]
        self.plot_containers[index] = (
            container_ref[0],
            container_ref[1],
            container_ref[2],
            new_handle,
        )

    def toggle_plot_visibility(self, index: int) -> None:
        """Toggle visibility of a plot."""
        self.plot_visible[index] = not self.plot_visible[index]
        container, plot_widget, min_btn, resize_handle = self.plot_containers[index]

        if self.plot_visible[index]:
            plot_widget.show()
            resize_handle.show()
            min_btn.setText("−")
        else:
            plot_widget.hide()
            resize_handle.hide()
            min_btn.setText("+")

    def on_freeze_toggled(self, checked: bool) -> None:
        """Handle freeze button toggle."""
        self.frozen = checked
        self.export_btn.setVisible(checked)
        if checked:
            self.freeze_btn.setText("Frozen")
        else:
            self.freeze_btn.setText("Freeze")

    def apply_autoscale(self) -> None:
        """Apply autoscaling based on current buffer data."""
        if not self.autoscale_enabled:
            return

        for i, plot in enumerate(self.plots):
            if not self.plot_visible[i]:
                continue

            y_data = self.sample_buffers[i]
            valid_data = y_data[y_data != 0] if np.any(y_data != 0) else y_data

            if len(valid_data) > 0:
                y_min, y_max = np.min(valid_data), np.max(valid_data)
                y_range = y_max - y_min
                y_padding = y_range * 0.1 if y_range > 0 else 0.5
                plot.setYRange(y_min - y_padding, y_max + y_padding)

    def rescale_to_first_10_percent(self) -> None:
        """Rescale plots based on first 10% of samples in buffer."""
        num_samples_to_use = max(1, self.buff_size // 10)

        for i, plot in enumerate(self.plots):
            if not self.plot_visible[i]:
                continue

            y_data = self.sample_buffers[i, :num_samples_to_use]
            valid_data = y_data[y_data != 0] if np.any(y_data != 0) else y_data

            if len(valid_data) > 0:
                y_min, y_max = np.min(valid_data), np.max(valid_data)
                y_range = y_max - y_min
                y_padding = y_range * 0.1 if y_range > 0 else 0.5
                plot.setYRange(y_min - y_padding, y_max + y_padding)

        print(
            f"Rescaled to first {num_samples_to_use} samples ({num_samples_to_use / self.buff_size * 100:.1f}% of buffer)"
        )

    def on_rescale(self) -> None:
        """Handle rescale button click."""
        self.rescale_to_first_10_percent()

    def update_data(self, values: np.ndarray) -> None:
        """Update all plots with new data."""
        if not self.frozen and len(values) == self.num_signals:
            if not self.autoscale_applied:
                self.sample_count += 1

                if self.sample_count > SKIP_INITIAL_SAMPLES:
                    self.autoscale_samples.append(values.copy())

                if len(self.autoscale_samples) == AUTOSCALE_SAMPLES:
                    print(
                        f"Autoscale enabled based on samples {SKIP_INITIAL_SAMPLES + 1}-{SKIP_INITIAL_SAMPLES + AUTOSCALE_SAMPLES}"
                    )
                    self.autoscale_applied = True

            self.sample_buffers = np.column_stack([values, self.sample_buffers[:, :-1]])

            for i, curve in enumerate(self.curves):
                curve.setData(self.x_axis, self.sample_buffers[i])

            if self.autoscale_enabled and self.autoscale_applied:
                self.apply_autoscale()

    def on_export_csv(self) -> None:
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
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export data: {e}")
