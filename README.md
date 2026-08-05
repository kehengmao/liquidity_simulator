# Liquid Raw

Liquid Raw is a real-time visualization demo for estimating global market
liquidity from traded volume. It transforms OHLCV candles into a time-price
field, models how liquidity is generated, retained, and swept across price
levels, and renders the resulting estimate beside a live candlestick chart.

The project is designed for MetaTrader 5 data on Windows. Its numerical core
uses NumPy and Numba, with an optional ahead-of-time (AOT) compiled CPython
extension. The visualization layer uses PySide6 and pyqtgraph.

In live operation, the online pipeline has demonstrated stable real-time data
updates at 200 ms intervals.

> This is a model-derived liquidity estimate, not an exchange order book. It is
> intended for visualization, research, and demonstration purposes only.

## Demo

![Liquid Raw real-time liquidity visualization](demo_screenshot.jpg)

## What it does

- Converts OHLCV candles into a 3D `[time, price, channel]` feature tensor.
- Distributes traded volume over each candle's price range.
- Represents body, shadow, and OHLC-point energy in separate channels.
- Applies a nonlinear volume-response curve to estimate liquidity attenuation.
- Traces historical energy through a transparency field to simulate liquidity
  persistence and sweeping.
- Maintains rolling time and price windows with circular buffers for real-time
  updates.
- Displays recent minimum, current, and maximum energy profiles next to the
  candlestick chart.
- Supports both live monitoring and deterministic historical replay.

## Processing pipeline

```text
MetaTrader 5 OHLCV
        |
        v
Adaptive price bins + candle anatomy
        |
        v
Time-price-channel tensor
        |
        +--> volume response / attenuation
        +--> transparency through history
        +--> body and OHLC energy flows
        |
        v
Global price-level liquidity estimate
        |
        v
PySide6 + pyqtgraph visualization
```

## Requirements

- Windows 10 or later
- 64-bit Python 3.10 recommended
- A locally installed MetaTrader 5 terminal
- An MT5 account with access to the requested symbol

The included `kernels_core.cp310-win_amd64.pyd` is built for 64-bit CPython
3.10 on Windows. On a different Python ABI, the engine automatically falls
back to Numba JIT compilation. The first JIT execution may take longer while
Numba compiles the kernels.

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Make sure the MetaTrader 5 desktop terminal is installed and can log in before
starting the demo.

## MT5 credentials

Credentials are never stored in the repository. You can provide them through
environment variables:

```powershell
$env:LIQUID_MT5_LOGIN = "your-login"
$env:LIQUID_MT5_PASSWORD = "your-password"
$env:LIQUID_MT5_SERVER = "your-broker-server"
```

If an environment variable is missing, the program asks for it interactively;
the password prompt does not echo its value.

## Run the demo

Start the application from the repository root:

```powershell
python liquidity_raw.py
```

At the configuration prompt, use one of these forms:

```text
Live:
SYMBOL MINUTES MAX_REFRACTION INCREMENTAL_BARS [INITIAL_BARS]

Replay:
SYMBOL MINUTES MAX_REFRACTION INCREMENTAL_BARS INITIAL_BARS REPLAY_BARS [STEP_DELAY]
```

Example live configuration:

```text
XAUUSD.s 1 0.9 60 10000
```

Example replay configuration:

```text
XAUUSD.s 1 0.9 60 1000 1500 0.0001
```

The symbol name is broker-specific. Use the exact name shown in your MT5
terminal.

### Configuration fields

| Field | Meaning |
| --- | --- |
| `SYMBOL` | Broker-specific MT5 symbol |
| `MINUTES` | Candle interval; supported mappings are 1, 5, 15, 30, and 60 |
| `MAX_REFRACTION` | Maximum attenuation coefficient used by the field model |
| `INCREMENTAL_BARS` | Recent bars appended one at a time during initialization |
| `INITIAL_BARS` | Initial history size; defaults to 9999 |
| `REPLAY_BARS` | Historical bars preloaded for replay mode |
| `STEP_DELAY` | Optional delay in seconds between replay steps; defaults to 0 |

## Chart controls

| Action | Control |
| --- | --- |
| Zoom time and price | Mouse wheel |
| Zoom time only | Ctrl + mouse wheel |
| Zoom price only | Shift + mouse wheel |
| Pan | Mouse drag |
| Reset automatic framing | Double-click |
| Toggle horizontal price guide | Left-click |

## Use the engine directly

`LiquidEngine` accepts a pandas DataFrame containing `open`, `high`, `low`,
`close`, and `volume` columns:

```python
from liba.fields import LiquidEngine

engine = LiquidEngine(tick_size=0.01, max_refraction=0.9)
engine.load_data(ohlcv_dataframe)

# Return the last 60 logical frames as [time, price].
energy = engine.get_total_energy((-60, 0))
```

On the first load, the engine derives an adaptive price-bin size from average
candle range. Later updates reuse the allocated time-price cube and remap the
price axis only when necessary.

## Build the AOT extension

The repository already includes a CPython 3.10 Windows build. To rebuild it for
the active compatible environment, install Microsoft C++ Build Tools and run:

```powershell
python -m liba.fields.kernel.compile
```

The generated platform-specific `kernels_core` extension is written to
`liba/fields/kernel/`. If the extension cannot be imported, `LiquidEngine`
continues with the Numba JIT implementation.

## Production standalone build

For a production-style full standalone build, install Nuitka and compile the
application with the required packages included:

```powershell
python -m pip install nuitka
nuitka --standalone --include-package=PySide6 --include-package=liba --include-package=MetaTrader5 --enable-plugin=pyside6 liquidity_raw.py
```

## Repository layout

```text
liquidity_raw.py                 Interactive entry point
liba/runner.py                   MT5 live/replay orchestration
liba/drawer.py                   PySide6 and pyqtgraph renderer
liba/fields/engine.py            Real-time rolling liquidity engine
liba/fields/kernel/kernels.py    Accelerated numerical kernels
liba/fields/kernel/compile.py    Numba AOT build script
liba/fields/static_field_demo.py Earlier static-field reference implementation
```

## Model limitations

- The estimate is inferred from candle geometry and traded volume; it does not
  observe pending limit orders or full market depth.
- Tick volume is used when MT5 does not provide nonzero real volume.
- Live values near a candle boundary can differ from settled replay values
  because late ticks may not yet be present in the local MT5 cache.
- Price binning deliberately trades sub-bin microstructure for bounded memory
  use and real-time performance.
- Output should not be treated as financial advice or an execution signal.
