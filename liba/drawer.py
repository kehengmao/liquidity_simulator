import sys
import os
import threading
import time
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
import pyqtgraph as pg
from collections import deque
import gc


class CandlestickItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self._rect = QtCore.QRectF()
        self.width = 0.6

        # 1. Pens draw candle borders.
        self._green_pen = pg.mkPen('g', width=0)
        self._red_pen = pg.mkPen('r', width=0)

        # 2. Brushes fill candle bodies; pyqtgraph accepts color strings directly.
        self._green_brush = pg.mkBrush('g')
        self._red_brush = pg.mkBrush('r')

    def updateData(self, data, ref_candle: 'CandlestickItem' = None):
        if not data:
            return

        self.data = data

        # Reuse the reference layer's bounds when one is available.
        if ref_candle is not None:
            # Matching bounds keep both items identical from the ViewBox perspective.
            new_rect = ref_candle.boundingRect().adjusted(-1, -1, 1, 1)
        else:
            # The historical layer computes its own bounds.
            x_coords = [d[0] for d in data]
            y_highs = [d[4] for d in data]
            y_lows = [d[3] for d in data]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_lows), max(y_highs)

            new_rect = QtCore.QRectF(
                min_x - self.width,
                min_y,
                (max_x - min_x) + self.width * 2,
                (max_y - min_y)
            )

        # Notify the scene only when geometry actually changes.
        if self._rect != new_rect:
            self.prepareGeometryChange()
            self._rect = new_rect

        self.data = data
        self.update()

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        for d in self.data:
            t, open_p, close_p, low_p, high_p = d

            # Switch the pen and brush together when candle direction changes.
            pen, brush = (self._green_pen, self._green_brush) if close_p >= open_p else (self._red_pen, self._red_brush)

            p.setPen(pen)
            p.setBrush(brush)

            # Draw the wick with ``t`` at its center.
            p.drawLine(QtCore.QLineF(t, float(low_p), t, float(high_p)))

            # Center the candle body around ``t``.
            rect = QtCore.QRectF(t - self.width / 2, float(open_p), self.width, float(close_p - open_p))
            p.drawRect(rect)

    def boundingRect(self):
        return self._rect

class TimeAxisItem(pg.AxisItem):
    """
    Map numeric candle indices to timestamps for X-axis labels.
    """
    def __init__(self, tick_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tick_map = tick_map  # Shared with VolumeProfileChart.idx_to_time_map.

    def tickStrings(self, values, scale, spacing):
        # pyqtgraph supplies numeric view coordinates.
        rst = [self._format_time(v) for v in values]
        # print(f'Triggered Time Str Rst: {len(rst)}')
        return rst

    def _format_time(self, v):
        idx = int(round(v))
        ts = self.tick_map.get(idx)
        if ts:
            time_str = pd.to_datetime(ts, unit='s').strftime('%H:%M')
            # print(f'Trigger: {time_str}')
            return time_str
        return ""

class VolumeProfileChart(pg.PlotWidget):
    def __init__(self, volume_dict, tick_size=0.5):
        super().__init__()
        print(f'Received Bin Size: {tick_size}')
        self.idx_to_time_map:dict = {}

        self.tick_size = tick_size
        self.all_data_history: deque | None = None
        self.last_candle_time = 0
        self.volume_dict = volume_dict  # low, newest, high
        self.auto_range_enabled = True

        # Enable hover tracking so the crosshair moves without a pressed button.
        self.setMouseTracking(True)
        self.following_mouse = False

        # 1. Main candlestick layer.
        self.candles = CandlestickItem()
        self.addItem(self.candles)

        self.last_candle = CandlestickItem()  # Dedicated live-candle layer.
        self.addItem(self.last_candle)

        # 2. Activate the right-hand price axis.
        self.showAxis('right')
        self.getPlotItem().getAxis('right').linkToView(self.getViewBox())
        self.getPlotItem().getAxis('right').showLabel(False)

        # 3. Configure the volume-profile overlay.
        self.vp_view = pg.ViewBox()
        self.scene().addItem(self.vp_view)
        self.vp_view.setXRange(0, 100)
        self.vp_view.setYLink(self.getViewBox())
        self.vp_view.setMouseEnabled(x=False, y=False)

        # 4. Initialize volume-profile bar layers.
        self.vp_bars_high = self._add_bar_graph(0.25)
        self.vp_bars_newest = self._add_bar_graph(0.5)
        self.vp_bars_low = self._add_bar_graph(1.0)

        # 5. Dashed price guide.
        self.price_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen((200, 200, 200), width=1, style=QtCore.Qt.PenStyle.DashLine),
            label='{value:0.2f}',
            labelOpts={
                'position': 0.95,
                'color': (200, 200, 200),
                'fill': (50, 50, 50, 200)
            }
        )
        self.price_line.hide()
        self.addItem(self.price_line)

        self.time_axis = TimeAxisItem(tick_map=self.idx_to_time_map, orientation='bottom')
        self.getPlotItem().setAxisItems({'bottom': self.time_axis})
        self.showAxis('bottom')
        self.time_axis.setTickSpacing(major=20, minor=5)

        bottom_axis = self.getPlotItem().getAxis('bottom')
        bottom_axis.setHeight(40)
        bottom_axis.show()

        self.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)

        # print("--- X-axis status ---")
        # print(f"Visible: {bottom_axis.isVisible()}")
        # print(f"Height: {bottom_axis.height()}")
        # print(f"Text color: {bottom_axis.textPen().color().name()}")
        # print(f"Axis color: {bottom_axis.pen().color().name()}")
        # print(f"Scene: {bottom_axis.scene()}")

        # 6. Keep the overlay geometry synchronized with the main view.
        self.getViewBox().sigResized.connect(self._update_vp_view_layout)
        self._update_vp_view_layout()

    def _add_bar_graph(self, solid_pct, color=(0, 150, 255)):
        # Initialize one volume-profile bar layer.
        solid = int(solid_pct * 255)
        r, g, b = color
        vp_bars = pg.BarGraphItem(
            x0=100, y=[], width=[], height=self.tick_size,
            brush=pg.mkBrush(r, g, b, solid),
            pen=None
        )
        self.vp_view.addItem(vp_bars)

        return vp_bars


    def put_kline_data(self, data_list):
        """
        Update the chart with a candlestick data sequence.

        Args:
            data_list: Nested lists in the form
                ``[[time, open, close, low, high], ...]``. ``time`` is a Unix
                timestamp in seconds and the remaining values are prices.
        """
        if not data_list:
            return

        self._update_kline_history(data_list)
        self._candles_update()
        self._refresh_volume_profile()

        if self.auto_range_enabled:
            self._auto_range()

    def update_latest_kline(self, last_bar_data):
        """Update the changing candle without touching historical data."""
        if not self.all_data_history:
            return
        if last_bar_data[0]!=self.last_candle_time:
            self.last_candle_time = last_bar_data[0]
            # self.last_candle.clear()

        # The live candle follows the last historical index.
        # If history occupies indices 0-99, the live candle uses index 100.
        last_idx = len(self.all_data_history)

        # Prepare one candle in plot coordinates.
        # last_bar_data: [time, open, close, low, high]
        single_plot_data = [[last_idx, last_bar_data[1], last_bar_data[2], last_bar_data[3], last_bar_data[4]]]

        # Update only the live layer.
        self.last_candle.updateData(single_plot_data, ref_candle=self.candles)

    def _candles_update(self):
        # Rebuild the indexed plot list for CandlestickItem.
        plot_data = []
        self.idx_to_time_map.clear()

        for i, data in enumerate(self.all_data_history):
            # Input: [timestamp, open, close, low, high]
            # Plot: [x_index, open, close, low, high]
            plot_data.append([i, data[1], data[2], data[3], data[4]])
            # Store the timestamp used by the bottom axis formatter.
            self.idx_to_time_map[i] = int(data[0])

        # Candle width is 0.6 for an index spacing of 1.0.
        self.candles.updateData(plot_data)

    def _update_kline_history(self, data_list):
        if self.all_data_history is None:
            self.all_data_history = deque(maxlen=len(data_list))
            self.all_data_history.extend(data_list)
        else:
            # print(f"History: {len(self.all_data_history)}, incoming: {len(data_list)}")

            last_recorded_time = self.all_data_history[-1][0]
            # Merge the incoming packet into the bounded history.
            for item in data_list:
                current_time = int(item[0])

                if current_time > last_recorded_time:
                    # A genuinely new candle is appended.
                    self.all_data_history.append(item)
                    last_recorded_time = current_time

                elif current_time == last_recorded_time:
                    # An update to the current candle replaces the last record.
                    self.all_data_history[-1] = item

                else:
                    # Discard stale packets, which commonly arrive after network delay.
                    continue

    def _refresh_volume_profile_single(self, vp_bars, vol_dict, max_v):

        # Extract all incoming volume-profile values.
        indices = vol_dict.keys()
        volumes = vol_dict.values()

        # Convert logical indices to real price coordinates.
        y_coords = [idx * self.tick_size for idx in indices]
        widths = [-(v / max_v) * 30 for v in volumes]

        # Submit the complete profile to the graphics layer.
        vp_bars.setOpts(
        x0=100,  # A scalar is broadcast across all Y coordinates.
        y=y_coords,
        width=widths,
        height=self.tick_size,
        brush=vp_bars.opts['brush'],  # Preserve the configured brush.
        pen=vp_bars.opts['pen']       # Preserve the configured pen.
        )

        # Keep the overlay X axis in a fixed percentage range.
        self.vp_view.setXRange(0, 100, padding=0)

    def _refresh_volume_profile(self):
        all_vals = []
        for key in ['high', 'newest', 'low']:
            sub_dict = self.volume_dict.get(key, {})
            if sub_dict:
                all_vals.extend(sub_dict.values())

        global_max_v = max(all_vals) if all_vals else 1

        self._refresh_volume_profile_single(self.vp_bars_low, self.volume_dict['low'], global_max_v)
        self._refresh_volume_profile_single(self.vp_bars_newest, self.volume_dict['newest'], global_max_v)
        self._refresh_volume_profile_single(self.vp_bars_high, self.volume_dict['high'], global_max_v)

        self.scene().update()

    def _update_vp_view_layout(self):
        """Keep the volume-profile overlay aligned with the main plot."""
        vb_rect = self.getViewBox().sceneBoundingRect()
        axis_rect = self.getPlotItem().getAxis('bottom').sceneBoundingRect()

        # print("--- Layout overlap check ---")
        # print(f"Main plot bounds: {vb_rect}")
        # print(f"Bottom-axis bounds: {axis_rect}")

        self.vp_view.setGeometry(vb_rect)

    def _auto_range(self):
        """
        Automatically frame the chart.

        The X axis shows the latest 100 candles, and the Y axis centers the
        visible candles with enough vertical padding.
        """
        if not self.all_data_history or len(self.all_data_history) == 0:
            return

        # 1. X-axis range.
        total_len = len(self.all_data_history)
        display_count = 100

        last_idx = total_len - 1
        x_min = last_idx - display_count
        x_max = last_idx + 20  # Leave room for the volume profile.

        # 2. Center visible candles at roughly half the plot height.
        start_idx = max(0, total_len - display_count)
        visible_data = list(self.all_data_history)[start_idx:]

        # Find the high and low of the visible candle window.
        k_lows = [d[3] for d in visible_data]
        k_highs = [d[4] for d in visible_data]

        if not k_lows or not k_highs:
            return

        v_min = min(k_lows)
        v_max = max(k_highs)
        price_range = v_max - v_min

        if price_range <= 0:
            price_range = v_min * 0.01  # Avoid a zero range for flat prices.

        # To occupy 50% of the height, candles need 25% padding on each side.
        padding = price_range * 0.5

        actual_min = v_min - padding
        actual_max = v_max + padding

        # 3. Apply the calculated view range.
        self.setXRange(x_min, x_max, padding=0)
        self.setYRange(actual_min, actual_max, padding=0)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            # Toggle the price guide.
            self.following_mouse = not self.following_mouse

            if self.following_mouse:
                # Use floating-point coordinates while following the pointer.
                pos = ev.position()
                # Let the ViewBox map scene coordinates into data coordinates.
                mouse_point = self.getViewBox().mapSceneToView(pos)

                self.price_line.setPos(mouse_point.y())
                self.price_line.show()
            else:
                self.price_line.hide()

        # Preserve the built-in pan and zoom behavior.
        super().mousePressEvent(ev)

    def wheelEvent(self, ev):
        """Handle wheel zooming under Qt 6."""
        self.auto_range_enabled = False
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        # Wheel up zooms in; wheel down zooms out.
        scale_fact = 0.9 if ev.angleDelta().y() > 0 else 1.1

        vb = self.getViewBox()

        if modifiers == Qt.KeyboardModifier.ControlModifier:
            # Zoom time only.
            vb.scaleBy(x=scale_fact, y=1.0)
        elif modifiers == Qt.KeyboardModifier.ShiftModifier:
            # Zoom price only.
            vb.scaleBy(x=1.0, y=scale_fact)
        else:
            # Zoom both axes by default.
            vb.scaleBy(x=scale_fact, y=scale_fact)

        ev.accept()

    def mouseMoveEvent(self, ev):
        if self.following_mouse:
            pos = ev.position()
            # Map directly through the ViewBox.
            mouse_point = self.getViewBox().mapSceneToView(pos)
            self.price_line.setPos(mouse_point.y())

        super().mouseMoveEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        # Double-click returns to automatic framing.
        self.auto_range_enabled = True
        # Apply the automatic range immediately.
        self._auto_range()
        ev.accept()

    def mouseDragEvent(self, ev):
        """Handle horizontal panning and vertical price-range adjustment."""
        # Any manual drag disables automatic framing.
        if ev.isStart():
            self.auto_range_enabled = False
            # print("Manual drag detected; automatic framing disabled")

        # Delegate the actual pan or zoom operation to pyqtgraph.
        super().mouseDragEvent(ev)


class Drawer(QtWidgets.QMainWindow):
    """Main window coordinating the UI thread and data worker thread."""
    sig_kline = QtCore.Signal(object, bool)
    sig_profile = QtCore.Signal(object)

    def __init__(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        super().__init__()
        self.chart: VolumeProfileChart | None = None
        self.volume_dict = {}
        self.tick_size = 0.5
        self._running = True

        self.sig_kline.connect(self._push_new_kline)
        self.sig_profile.connect(self._push_new_profile)

    def update_tick_size(self, tick_size):
        self.tick_size = tick_size

    def create_chart(self):
        new_chart = VolumeProfileChart(volume_dict = self.volume_dict, tick_size=self.tick_size)
        self.setCentralWidget(new_chart)
        self.resize(1000, 600)

        if self.chart is not None:
            new_chart.all_data_history = self.chart.all_data_history
            self.chart.deleteLater()
            gc.collect()

        self.chart = new_chart

    # Worker-facing methods emit signals instead of drawing across threads.
    def push_new_kline(self, df: pd.DataFrame, is_last: bool = False):
        self.sig_kline.emit(df, is_last)

    def push_new_profile(self, profile):
        self.sig_profile.emit(profile)

    def _push_new_kline(self, df: pd.DataFrame, is_last: bool = False):
        """
        Update candlestick data.

        Args:
            df: DataFrame containing ``time``, ``open``, ``close``, ``low``,
                and ``high`` columns. Time values are Unix seconds.
            is_last: Whether the row represents the currently changing candle.
        """
        if self.chart is None:
            self.create_chart()

        if not is_last:
            data = df[['time', 'open', 'close', 'low', 'high']].values.tolist()

            # if not data:
            #     print("The renderer received an empty candle batch")
            # else:
            #     print(f"The renderer received {len(data)} candles")

            self.chart.put_kline_data(data)
        else:
            data = df[['time', 'open', 'close', 'low', 'high']].values.tolist()[0]
            self.chart.update_latest_kline(data)

    def _push_new_profile(self, profile):
        self.volume_dict['low'] = profile['low']
        self.volume_dict['high'] = profile['high']
        self.volume_dict['newest'] = profile['newest']

    def start(self, logic_func=None):
        self.show()
        if logic_func:
            t = threading.Thread(target=logic_func, args=(self,), daemon=True)
            t.start()
        sys.exit(self.app.exec())

    def stop(self):
        """
        Stop the process immediately, including the worker thread.

        This bypasses normal interpreter cleanup and should only be used by the
        standalone visualization demo.
        """
        print("Stop requested; terminating the process.")
        self._running = False

        # This intentionally bypasses destructors and normal cleanup hooks.
        os._exit(0)


if __name__ == '__main__':
    # Standalone renderer demo running its producer in a worker thread.
    def my_logic(drawer):
        """Push 20 simulated data points, then stop the demo."""
        history_k = []

        for i in range(20):
            if not drawer._running: break

            print(f"Worker is pushing sample {i+1}/20...")

            # Generate a synthetic candle and matching energy profile.
            price = 100 + np.random.uniform(-5, 5)
            vol = np.random.randint(10, 100)
            history_k.append([int(time.time()) + i * 60, price, price + 1, price - 1, price + 2])

            # Send data to the UI thread.
            frame = pd.DataFrame(
                history_k,
                columns=['time', 'open', 'close', 'low', 'high'],
            )
            drawer.push_new_kline(frame)
            price_idx = int(round(price / drawer.tick_size))
            profile = {
                'low': {price_idx - 1: vol * 0.5},
                'newest': {price_idx: vol},
                'high': {price_idx + 1: vol * 1.5},
            }
            drawer.push_new_profile(profile)

            time.sleep(0.5)

        print("All 20 samples were pushed; stopping the demo.")
        drawer.stop()

    dr = Drawer()
    dr.update_tick_size(0.5)

    # Start the UI and the worker function.
    dr.start(logic_func=my_logic)
