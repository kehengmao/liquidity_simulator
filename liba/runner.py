from dataclasses import dataclass, field
from getpass import getpass
import os

import pandas as pd
import numpy as np
import time
from datetime import datetime

import MetaTrader5 as mt5


from .fields import LiquidEngine
from .drawer import Drawer


_is_test = False


def _read_mt5_credentials() -> tuple[str, str, str]:
    """Read MT5 credentials from the environment or secure interactive prompts."""
    account = os.getenv("LIQUID_MT5_LOGIN") or input("MT5 login: ").strip()
    password = os.getenv("LIQUID_MT5_PASSWORD") or getpass("MT5 password: ")
    server = os.getenv("LIQUID_MT5_SERVER") or input("MT5 server: ").strip()
    return account, password, server


def mt5_connect(account: str, password: str, server: str):
    if not mt5.initialize(login=int(account), password=password, server=server):
        print(f"Connection failed. MT5 error: {mt5.last_error()}")
        return False
    print("MT5 login successful.")
    return True


def quick_mt5_setup():
    print("--- Liquid MT5 configuration (live and replay modes) ---")
    print("Mouse: wheel zooms both axes; Ctrl+wheel zooms time; Shift+wheel zooms price.")
    print("Drag to pan, double-click to reset the view, and click to toggle the price guide.")
    print("Live:   SYMBOL MINUTES MAX_REFRACTION INCREMENTAL_BARS [INITIAL_BARS]")
    print("Replay: SYMBOL MINUTES MAX_REFRACTION INCREMENTAL_BARS INITIAL_BARS REPLAY_BARS [STEP_DELAY]")

    raw_input = input("\nEnter configuration: ")
    data = raw_input.replace(',', ' ').split()

    if len(data) < 4:
        print("Error: expected at least four configuration values.")
        return None

    account, password, server = _read_mt5_credentials()
    target_symbol, minutes, max_force, num_incs_bars = data[:4]

    mode = "realtime"
    backtest_count = 0
    backtest_sleep = 0.0  # Run replay at full speed by default.
    init_num = 9999

    if len(data) >= 5:
        init_num = int(data[4])

    if len(data) >= 6:
        mode = "backtest"
        backtest_count = int(data[5])

    if len(data) >= 7:
        backtest_sleep = float(data[6])  # Simulated delay between replay steps.
        print(f"Replay delay: {backtest_sleep} seconds per step")

    # 1. Connect to MT5.
    if not mt5_connect(account, password, server):
        return None

    # 2. Validate and activate the requested symbol.
    if not mt5.symbol_select(target_symbol, True):
        print(f"Error: symbol [{target_symbol}] is unavailable.")
        mt5.shutdown()
        return None

    # 3. Create the data loader with the selected operating mode.
    try:
        loader = MT5MinuteDataLoader(
            target=target_symbol,
            interval_nums=int(minutes),
            mode=mode,
            backtest_count=backtest_count,
            backtest_sleep = backtest_sleep
        )
        rates = mt5.copy_rates_from_pos(target_symbol, mt5.TIMEFRAME_M1, 0, init_num)
        if rates is None:
            print("Unable to retrieve market data.")

        app = LiquidWrapper(loader, init_num, float(max_force), int(num_incs_bars))

        # Replay data is resident in memory, so the MT5 connection can close.
        if mode == "backtest":
            print("Historical data loaded; closing the MT5 connection.")
            mt5.shutdown()

        mode_label = "Replay" if mode == "backtest" else "Live"
        print(f"{mode_label} loader ready.")
        return app

    except Exception as e:
        print(f"Initialization failed: {e}")
        return None

MIN_MAP = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1
        }

@dataclass
class MT5MinuteDataLoader:
    target: str
    interval_nums: int = 1
    watch_sleep_time: float = 0.1
    mode: str = "realtime"  # Either "realtime" or "backtest".
    backtest_count: int = 2000  # Number of bars to preload in replay mode.
    backtest_sleep: float = 1.0

    last_time: int = 0

    trigger_count: int = 0  # Number of completed-bar callback invocations.

    def __post_init__(self):
        """Map the requested minute interval to an MT5 timeframe constant."""
        self.tf_constant = MIN_MAP.get(self.interval_nums, mt5.TIMEFRAME_M1)

        self.info = mt5.symbol_info(self.target)

        self._adjust_time_offset()

        if self.mode == "backtest":
            # Preload all bars needed for deterministic replay.
            print(f"--- Preparing replay: preloading {self.backtest_count} bars ---")
            self.backtest_pool = self._fetch(self.backtest_count)
            self.cursor = 0
        else:
            self.info = mt5.symbol_info(self.target)

    def _adjust_time_offset(self):
        tick = mt5.symbol_info_tick(self.target)
        if tick is None:
            self.time_zone = 0
            print("Warning: broker time is unavailable; timestamps will not be adjusted.")
            return

        broker_time_int = tick.time  # Broker timestamp in seconds.
        local_time_int = int(datetime.now().timestamp())  # Local timestamp in seconds.

        # Estimate the broker-to-local offset in whole hours.
        diff_seconds = local_time_int - broker_time_int
        diff_hours = round(diff_seconds / 3600)

        print(f"Estimated broker-to-local time offset: {diff_hours} hours")
        self.time_zone = diff_hours

    def fetch(self, count: int, is_step: bool = False, is_fetch_close: bool = False):
        """Fetch bars through one interface for both live and replay modes."""
        if self.mode == "realtime":
            return self._fetch(count)
        else:
            if is_step:
                step = count
                if is_fetch_close:
                    step -=1
                self._backtest_step(step)
            time.sleep(self.backtest_sleep)
            # Present replay data through the same interface as live data.
            if self.cursor >= len(self.backtest_pool):
                print("Replay complete.")
                return None
            # Return the requested window ending at the current replay cursor.
            start_idx = max(0, self.cursor - count + 1)
            mock_slice = self.backtest_pool.iloc[start_idx : self.cursor + 1]

            return mock_slice

    def fetch_close(self, count: int, is_step: bool = False):
        """Fetch the most recent completed bars."""
        df = self.fetch(count+1, is_step = is_step, is_fetch_close = True)
        return df[:-1]

    def _fetch(self, count: int):
        """Fetch the latest bars directly from MT5."""
        rates = mt5.copy_rates_from_pos(self.target, self.tf_constant, 0, count)
        if rates is None:
            print(f"Market-data request failed: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        if 'real_volume' in df.columns and df['real_volume'].sum() > 0:
            df['volume'] = df['real_volume']
        else:
            df['volume'] = df['tick_volume']

        # Align broker timestamps with the local display timezone.
        df.loc[:, 'time'] = df['time'].astype(int) + self.time_zone * 3600
        return df

    def _backtest_step(self, count: int):
        self.cursor += count

    def watch(self, callback_func, callback_func_changing):
        """
        Monitor completed and changing bars in live or replay mode.

        Live/replay consistency:
        1. Live mode polls the timestamp at MT5 array index 0. When it
           changes, ``fetch_close(1)`` returns the completed bar at index 1.
        2. Replay mode advances a cursor to reproduce the same transition.
        3. Both modes expose the completed bar immediately before the current
           reference point.

        The only expected difference is the settlement gap. At a live boundary
        such as 10:01:00.05, a few ticks may not yet have reached the local MT5
        cache. Live values can therefore reflect an in-flight snapshot, while
        replay data contains the settled historical bar.

        Args:
            callback_func: Called once when a new completed bar is confirmed.
            callback_func_changing: Called on each poll for the changing bar.
        """
        print(f"Monitoring {self.target} ({self.interval_nums} min)...")
        try:
            while True:
                # Only the newest timestamp is needed to detect a transition.
                res = self.fetch(1)
                if res is not None and len(res) > 0:

                    current_time = int(res['time'].iloc[0])

                    # A newer timestamp indicates that a bar has completed.
                    if current_time > self.last_time:
                        self.trigger_count += 1

                        event_time = time.strftime(
                            '%Y-%m-%d %H:%M:%S',
                            time.localtime(res['time'].iloc[0]),
                        )
                        print(f"[{self.target}] completed bar #{self.trigger_count} | time: {event_time}")

                        callback_func()
                        self.last_time = current_time

                    callback_func_changing()

                time.sleep(self.watch_sleep_time)
        except KeyboardInterrupt:
            print("Monitoring stopped.")

@dataclass
class LiquidWrapper:
    mt5_loader: MT5MinuteDataLoader
    init_fetch_num: int
    max_force: float
    num_incs_bars: int
    price_energy_dict: dict = field(default_factory=dict)
    _inited: bool = False
    _first_push: bool = True

    def __post_init__(self):
        print(f'Original Tick Size: {self.mt5_loader.info.trade_tick_size}')
        self.engine = LiquidEngine(self.mt5_loader.info.trade_tick_size, max_refraction=self.max_force)
        self.drawer = Drawer()

    def update(self, drawer: Drawer):
        self.mt5_loader.watch(self._update, self._update_last_kline)

    def _update(self):
        """Load older bars in bulk, then append recent bars incrementally."""
        if not self._inited:

            df = self.mt5_loader.fetch_close(self.init_fetch_num, is_step = True)

            if df is not None and not df.empty:
                self._inited = True
                self.engine.load_data(df[:-self.num_incs_bars])
                self.drawer.update_tick_size(self.engine.bin_size)

                df_tail = df[-self.num_incs_bars:]
                for i in range(len(df_tail)):
                    self.engine.load_data(df_tail.iloc[i:i+1])

                self._push_to_drawer(df)
        else:
            df = self.mt5_loader.fetch_close(1, is_step = True)

            if df is not None and not df.empty:
                # print(f"Fetched {len(df)} bars")
                self.engine.load_data(df)
                self._push_to_drawer(df)
            else:
                print("Warning: No data fetched from MT5.")



    def _push_to_drawer(self, df: pd.DataFrame):
        # self.plot_energy_with_candles(self.engine.tick_size)

        energy_array = self.engine.get_total_energy((-self.num_incs_bars, 0))  # [time, price]
        print(f'Shape of Energy: {energy_array.shape}')
        high = np.max(energy_array, axis=0) # [P]
        newest = energy_array[-1,:] # [P]
        low = np.min(energy_array, axis=0) # [P]

        current_limit = energy_array.shape[1]

        if self._first_push:
            self._first_push = False
            p_min, p_max = 0, current_limit
        else:
            p_min = max(self.engine._last_p_min_tick - self.engine.p_min_tick - self.engine.ticks_per_bin*2, 0)
            p_max = min(self.engine._last_p_max_tick - self.engine.p_min_tick + self.engine.ticks_per_bin*2, current_limit)

        # price_indices = np.arange(self.engine.p_min_tick, self.engine.p_min_tick + current_limit) * self.engine.ticks_per_bin
        price_indices = np.arange(self.engine.p_min_tick, self.engine.p_min_tick + current_limit)
        print(f"Price-index range: {price_indices.min()} to {price_indices.max()}")
        _dict = {}
        _dict['high'] = dict(zip(price_indices, high))
        _dict['low'] = dict(zip(price_indices, low))
        _dict['newest'] = dict(zip(price_indices, newest))

        # Legacy reference: ref_base = p_min + self.engine.p_min_tick

        self.drawer.push_new_profile(_dict)

        self.drawer.push_new_kline(df)

    def _update_last_kline(self):
        df = self.mt5_loader.fetch(1)
        self.drawer.push_new_kline(df, is_last = True)



