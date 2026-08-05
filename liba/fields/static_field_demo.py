
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit, prange


@njit(nogil=True, parallel=True, cache=False)
def _numba_build_3d_cube_full_anatomy(
    ohlcv: np.ndarray,      # Shape: [N, 5]
    p_min_tick: int,        # Baseline offset
    tick_size: float,       # Price increment
    out_cube: np.ndarray    # Shape: [N, price_range, 11]
):
    """
    Map raw OHLCV data into a 3D time-price feature tensor in place.

    Args:
        ohlcv: ``[N, 5]`` array ordered as open, high, low, close, volume.
        p_min_tick: Tick index of the global minimum price.
        tick_size: Price increment, such as 0.01 or 0.5.
        out_cube: Preallocated ``[N, price_range, 11]`` tensor. Channels are
            absolute volume density; four OHLC anchors; solid body; upper,
            lower, and combined shadows; full high-low coverage; and volume
            ratio normalized by the global maximum.
    """
    n = ohlcv.shape[0]

    # 1. Find the global volume maximum with a Numba-friendly O(N) scan.
    v_max = 0.0
    for k in range(n):
        if ohlcv[k, 4] > v_max:
            v_max = ohlcv[k, 4]

    # Guard against division by zero.
    safe_v_max = v_max if v_max > 0 else 1.0

    # 2. Build each independent candle frame in parallel.
    for i in prange(n):
        o, h, l, c, v = ohlcv[i]

        # Volume relative to the global peak.
        v_ratio = v / safe_v_max

        # 2.1 Map prices to aligned indices.
        o_idx = int(round(o / tick_size)) - p_min_tick
        h_idx = int(round(h / tick_size)) - p_min_tick
        l_idx = int(round(l / tick_size)) - p_min_tick
        c_idx = int(round(c / tick_size)) - p_min_tick

        # 2.2 Calculate body and full-range slices.
        s_min, s_max = min(o_idx, c_idx), max(o_idx, c_idx)
        total_range_slice = slice(l_idx, h_idx + 1)
        solid_range_slice = slice(s_min, s_max + 1)

        # 2.3 Fill all 11 feature channels.

        # Number of price ticks used for density calculations.
        num_ticks = h_idx - l_idx + 1

        if num_ticks > 0:
            # C0: absolute volume-energy layer.
            out_cube[i, total_range_slice, 0] = v / num_ticks
            # C10: distribute normalized volume ratio across the range.
            out_cube[i, total_range_slice, 10] = v_ratio / num_ticks

        # C1-C4: OHLC anchor layers.
        out_cube[i, o_idx, 1] = 1.0  # Open Point
        out_cube[i, c_idx, 2] = 1.0  # Close Point
        out_cube[i, h_idx, 3] = 1.0  # High Point
        out_cube[i, l_idx, 4] = 1.0  # Low Point

        # C5: solid-body layer.
        out_cube[i, solid_range_slice, 5] = 1.0

        # C6: upper-shadow layer.
        if h_idx > s_max:
            out_cube[i, s_max + 1 : h_idx + 1, 6] = 1.0
            out_cube[i, s_max + 1 : h_idx + 1, 8] = 1.0

        # C7: lower-shadow layer.
        if l_idx < s_min:
            out_cube[i, l_idx : s_min, 7] = 1.0
            out_cube[i, l_idx : s_min, 8] = 1.0

        # C9: complete high-low coverage layer.
        out_cube[i, total_range_slice, 9] = 1.0

# @njit(nogil=True, parallel=True, cache=False)
# def _numba_fast_refraction_calc(v_ratio_2d: np.ndarray):
#     """
#     Calculate the attenuation matrix in parallel.
#     """
#     n, p = v_ratio_2d.shape
#     out = np.empty((n, p), dtype=np.float32)

#     for i in prange(n):
#         for j in range(p):
#             v = v_ratio_2d[i, j]
#             # Inverted-U sweep-efficiency curve.
#             if v > 0.001:
#                 # Numba implementation of 4v(1-v).
#                 out[i, j] = 4.0 * v * (1.0 - v)
#             else:
#                 out[i, j] = 0.0
#     return out

PI_f32 = np.float32(3.141592653589793)

@njit(nogil=True, parallel=True, cache=False)
def _numba_fast_refraction_calc(v_ratio_2d: np.ndarray):
    """
    Calculate attenuation in parallel with a sine-response prior.

    This uses ``sin(pi * v)`` instead of ``4v(1-v)``.
    """
    n, p = v_ratio_2d.shape
    out = np.empty((n, p), dtype=np.float32)

    # PI is precomputed in single precision.

    for i in prange(n):
        for j in range(p):
            v = v_ratio_2d[i, j]
            # sin(pi*v) is zero at both boundaries; 0.001 defines a dead zone.
            if 0.001 < v < 0.999:
                # Numba dispatches sine to its optimized math implementation.
                out[i, j] = np.sin(PI_f32 * np.float32(v))
            else:
                out[i, j] = 0.0
    return out


@njit(nogil=True, parallel=True, cache=False)
def _numba_energy_flow_decoupled(cube_full: np.ndarray,
                                 time_features: np.ndarray,
                                 idx_all_body: int,
                                 idx_open: int,
                                 idx_close: int,
                                 idx_high: int,
                                 idx_low: int):
    """
    Split energy flow into five channels with shared maximum normalization.

    Returns ``[N, price, 5]`` ordered as body, open, close, high, and low.
    """
    n, p, _ = cube_full.shape
    # Allocate five output channels.
    out = np.zeros((n, p, 5), dtype=np.float32)

    for i in prange(n):
        v_base = time_features[i, 0]      # Volume
        v_intensity = time_features[i, 2]  # Enhanced OHLC energy.

        for j in range(p):
            # Channel 0: base energy across the full body.
            if cube_full[i, j, idx_all_body] > 0:
                out[i, j, 0] = v_base

            # Channels 1-4: assign intensity energy at OHLC points.
            if cube_full[i, j, idx_open] > 0:
                out[i, j, 1] = v_intensity

            if cube_full[i, j, idx_close] > 0:
                out[i, j, 2] = v_intensity

            if cube_full[i, j, idx_high] > 0:
                out[i, j, 3] = v_intensity

            if cube_full[i, j, idx_low] > 0:
                out[i, j, 4] = v_intensity

    # Find the global maximum across all five channels.
    v_max = out.max()

    # Normalize all channels with the shared maximum.
    if v_max > 1e-9:
        for i in prange(n):
            for j in range(p):
                for k in range(5):
                    out[i, j, k] /= v_max

    return out

@njit(nogil=True, parallel=True, cache=False)
def _numba_calc_transparency_map(all_body_plane: np.ndarray,
                                 refraction_plane: np.ndarray):
    """
    Precompute a global ``[N, P]`` transparency map in the range ``[0, 1]``.

    Each value represents the transparency remaining toward the newest frame.
    """
    n, p = all_body_plane.shape
    t_map = np.ones((n, p), dtype=np.float32)

    for j in prange(p):
        curr_trans = 1.0
        # Scan from the newest frame toward the oldest.
        for i in range(n - 1, -1, -1):
            # Energy does not pass through its own frame, only newer frames.
            t_map[i, j] = curr_trans

            # Update accumulated attenuation for the next older frame.
            if all_body_plane[i, j] > 0:
                curr_trans *= refraction_plane[i, j]

            # Stop early when older contributions become negligible.
            if curr_trans < 1e-4:
                # All remaining older frames receive zero transparency.
                for k in range(i - 1, -1, -1):
                    t_map[k, j] = 0.0
                break

    return t_map



@dataclass
class LiquidEngine:
    """
    Time-Price Field (LiquidEngine):
    Convert financial time series into a 3D feature tensor.
    """
    # Core tensor: [time, price, feature].
    cube: Optional[np.ndarray] = None

    # Time-domain feature matrix: [time, 3].
    time_square: Optional[np.ndarray] = None
    price_squre: Optional[np.ndarray] = None

    # Price-axis baseline offset.
    p_min_tick: int = 0
    tick_size: float = 0.01
    # Original timestamp index.
    timeline: pd.Index = field(default_factory=lambda: pd.Index([]))

    # Base 11-channel feature mapping.
    feature_map: dict = field(default_factory=lambda: {
        "volume": 0,         # Absolute volume density.
        "open_pt": 1,        # Opening-price point.
        "close_pt": 2,       # Closing-price point.
        "high_pt": 3,        # High-price point.
        "low_pt": 4,         # Low-price point.
        "solid_body": 5,     # Solid-body range.
        "upper_shadow": 6,   # Upper shadow.
        "lower_shadow": 7,   # Lower shadow.
        "all_shadows": 8,    # Combined shadows.
        "all_body": 9,       # Complete high-low coverage.
        "vol_ratio": 10      # Volume relative to the global maximum.
    })

    # Time-domain feature mapping.
    time_feature_map: dict = field(default_factory=lambda: {
        "volume": 0,              # Raw volume.
        "body_length": 1,         # High-low length in ticks.
        "intensity": 2            # Length multiplied by volume.
    })

    price_feature_map: dict = field(default_factory=lambda: {
        "ef_body_refr_sum": 0,   # Attenuated energy accumulated across bodies.
        "ef_open_refr_sum": 1,   # Residual influence of historical opens.
        "ef_close_refr_sum": 2,  # Residual influence of historical closes.
        "ef_high_refr_sum": 3,   # Residual resistance near historical highs.
        "ef_low_refr_sum": 4     # Residual support near historical lows.
    })

    def load_data(self, df: pd.DataFrame, tick_size: float):
        """Build the 11 base channels and inject derived field features."""
        self.tick_size = tick_size
        self.timeline = df.index

        # 1. Calculate price boundaries and tensor height.
        p_min = df['low'].min()
        p_max = df['high'].max()
        self.p_min_tick = int(np.floor(p_min / self.tick_size))
        p_max_tick = int(np.ceil(p_max / self.tick_size))
        price_range = p_max_tick - self.p_min_tick + 1

        # 2. Convert input to an efficient OHLCV NumPy layout.
        ohlcv_raw = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float64)

        # 3. Allocate the 11 base channels. Injection methods expand to 23.
        self.cube = np.zeros((len(df), price_range, 11), dtype=np.float32)

        # 4. Fill the base channels.
        _numba_build_3d_cube_full_anatomy(
            ohlcv_raw,
            self.p_min_tick,
            self.tick_size,
            self.cube
        )

        print("LiquidEngine base field built with 11 channels.")
        print(f"Price range: {self.to_real_price(0):.2f} -> {self.to_real_price(price_range-1):.2f}")

        # 5. Run the derived-feature pipeline: 11 -> 13 -> 18 -> 23 channels.
        self.load_time_feature(df)
        self.inject_refraction_filter()
        self.inject_transparency_map()
        self.inject_energy_flow()
        self.inject_refracted_energy_flow()

        print(f">>> LiquidEngine field complete with 23 channels: {self.cube.shape}")

    def load_time_feature(self, df: pd.DataFrame) -> None:
        """Extract time-series features from the DataFrame and generated cube."""
        if self.cube is None:
            print("Error: build the cube before calculating body length.")
            return

        # Allocate the [N, 3] time-feature matrix.
        n_steps = len(df)
        self.time_square = np.zeros((n_steps, len(self.time_feature_map)), dtype=np.float32)

        # Feature 1: raw volume.
        vol_data = df['volume'].values

        # Feature 2: body length projected from the cube for tick consistency.
        body_idx = self.feature_map["all_body"]
        body_lengths = np.sum(self.cube[:, :, body_idx], axis=1)

        # Feature 3: body-volume intensity.
        bv_efficiency = body_lengths * vol_data

        # Fill the feature matrix.
        self.time_square[:, 0] = vol_data
        self.time_square[:, 1] = body_lengths
        self.time_square[:, 2] = bv_efficiency

        print(f">>> Time features loaded: {self.time_square.shape}")

    def inject_refraction_filter(self):
        """Append the ``refraction_intensity`` channel to the cube."""
        if self.cube is None:
            print("Error: call load_data before injecting derived features.")
            return

        # 1. Extract the [N, price] volume-ratio plane.
        v_idx = self.feature_map["vol_ratio"]
        v_ratio_plane = self.cube[:, :, v_idx]

        # 2. Calculate attenuation/sweep efficiency in parallel.
        refraction_plane = _numba_fast_refraction_calc(v_ratio_plane)

        # 3. Add a singleton channel dimension and concatenate.
        self.cube = np.concatenate([
            self.cube,
            refraction_plane[:, :, np.newaxis]
        ], axis=-1)

        # 4. Update the feature mapping.
        new_idx = self.cube.shape[2] - 1
        self.feature_map["refraction_intensity"] = new_idx

        print(f">>> Attenuation channel injected at {new_idx}; features: {len(self.feature_map)}")

    def inject_transparency_map(self):
        """Inject a global transparency map derived from attenuation."""
        if "refraction_intensity" not in self.feature_map or "all_body" not in self.feature_map:
            print("Error: refraction_intensity and all_body are required.")
            return

        # 1. Extract the required base planes.
        all_body_plane = self.cube[:, :, self.feature_map["all_body"]]
        refraction_plane = self.cube[:, :, self.feature_map["refraction_intensity"]]

        # 2. Accumulate transparency from newest to oldest with Numba.
        t_map = _numba_calc_transparency_map(all_body_plane, refraction_plane)

        # 3. Append the map to the cube.
        self.cube = np.concatenate([
            self.cube,
            t_map[:, :, np.newaxis]
        ], axis=-1)

        # 4. Update the feature mapping.
        new_idx = self.cube.shape[2] - 1
        self.feature_map["transparency_map"] = new_idx

        print(f">>> Transparency map injected at {new_idx}.")

    def inject_energy_flow(self):
        """Inject the decoupled five-channel energy-flow field."""
        if self.cube is None or self.time_square is None:
            print("Error: the cube and time-feature matrix are required.")
            return

        # The accelerated function returns [N, price, 5].
        energy_channels = _numba_energy_flow_decoupled(
            self.cube,
            self.time_square,
            idx_all_body=self.feature_map["all_body"],
            idx_open=self.feature_map["open_pt"],
            idx_close=self.feature_map["close_pt"],
            idx_high=self.feature_map["high_pt"],
            idx_low=self.feature_map["low_pt"]
        )

        # Append the already-3D channels directly.
        self.cube = np.concatenate([self.cube, energy_channels], axis=-1)

        # Update feature_map with the appended channel indices.
        ef_names = ["ef_body", "ef_open", "ef_close", "ef_high", "ef_low"]

        # Find the first newly appended channel.
        start_idx = self.cube.shape[2] - 5
        for i, name in enumerate(ef_names):
            self.feature_map[name] = start_idx + i

        print(f">>> Decoupled energy channels injected: {ef_names}")
        print(f">>> Feature count: {len(self.feature_map)}")


    def inject_refracted_energy_flow(self):
        """
        Multiply five energy channels by transparency to obtain filtered energy.

        ``ef_refracted = ef_original * transparency_map``
        """
        if "transparency_map" not in self.feature_map:
            print("Error: inject transparency_map first.")
            return

        # 1. Expand the map to [N, price, 1] for broadcasting.
        t_map = self.cube[:, :, self.feature_map["transparency_map"]][:, :, np.newaxis]

        # 2. Resolve the five raw energy-channel indices.
        ef_names = ["ef_body", "ef_open", "ef_close", "ef_high", "ef_low"]
        ef_indices = [self.feature_map[name] for name in ef_names]

        # 3. Multiply [N, price, 5] by [N, price, 1].
        refracted_energy = self.cube[:, :, ef_indices] * t_map

        # 4. Append the filtered channels.
        self.cube = np.concatenate([self.cube, refracted_energy], axis=-1)

        # 5. Update the feature mapping.
        new_ef_names = [f"{name}_refr" for name in ef_names]
        start_idx = self.cube.shape[2] - 5
        for i, name in enumerate(new_ef_names):
            self.feature_map[name] = start_idx + i

        print(f">>> Filtered energy channels injected: {new_ef_names}")
        print(f">>> Feature count: {len(self.feature_map)}")

        self.collapse_energy_to_price_square()

    def collapse_energy_to_price_square(self):
        """
        Collapse time-price energy along the time axis into ``price_squre``.
        """
        # 1. Resolve the five filtered channel indices.
        ef_refr_names = ["ef_body_refr", "ef_open_refr", "ef_close_refr", "ef_high_refr", "ef_low_refr"]
        ef_indices = [self.feature_map[name] for name in ef_refr_names]

        # 2. Extract [time, price, 5] data.
        refracted_data = self.cube[:, :, ef_indices]

        # 3. Sum along time to produce [price, 5].
        collapsed_data = np.sum(refracted_data, axis=0)

        # 4. Store the collapsed result.
        self.price_squre = collapsed_data

        print(f">>> Energy collapsed into price_squre: {self.price_squre.shape}")


    def to_real_price(self, tick_idx: int) -> float:
        """Convert a cube price index back to a real price."""
        return (tick_idx + self.p_min_tick) * self.tick_size

    def to_tick_idx(self, price: float) -> int:
        """Convert a real price to a cube price index."""
        return int(round(price / self.tick_size)) - self.p_min_tick

    @property
    def raw_energy_profile(self):
        """
        Return the aggregate energy profile in the price domain.

        The five ``price_squre`` channels are summed into one value per price
        level, producing a one-dimensional NumPy array.
        """
        return np.sum(self.price_squre, axis=1)

    def wrap_result(self):
        """
        Package the energy profile and price-axis metadata.

        The returned dictionary contains ``energy`` (one value per price
        level), ``p_min`` (baseline logical tick), and ``size`` (tick size).
        """
        return {
        "energy": self.raw_energy_profile,
        "p_min": self.p_min_tick,
        "size": self.tick_size
    }
