# Marana-X native SDK3 wrapper: benchmark

Wrapper under test: `src/pymodaq_plugins_fab1b/hardware/andor_sdk3/andor_sdk3.py` (`AndorSDK3Camera`).

## Setup

- Camera: MARANA-4BN1X-SOM, serial CSC-00907, firmware 22.3.8.12400, USB
- Library: `C:\Program Files\Andor SOLIS\atcore.dll`
- Python env: `expdev2` (numpy 2.4.6)
- Gain mode: "Fastest frame rate (12-bit)", readout 200 MHz, internal trigger, rolling shutter
- Settings per case: minimum exposure, maximum FrameRate, 32 buffers
- Each case ran for 5 s, 3 independent runs (new process each)
- Consumer loop per frame: `wait_frame` -> `frame_view(...).copy()` -> `requeue`
  (the copy is required before requeueing, so it is included in the cost)

## Throughput

| Case | SDK FrameRate (fps) | Measured fps, runs 1/2/3 | Mean (fps) | % of SDK limit |
|---|---|---|---|---|
| Full 2048x2048, Mono16 | 47.50 | 44.86 / 44.80 / 44.74 | 44.80 | 94% |
| Full 2048x2048, Mono12Packed | 47.50 | 41.53 / 42.24 / 42.46 | 42.08 | 89% |
| ROI 2048x256, Mono16 | 372.0 | 360.58 / 360.52 / 360.23 | 360.44 | 97% |
| ROI 2048x16, Mono16 | 4430.3 | 4432.18 / 4431.31 / 4432.13 | 4431.87 | 100% |
| ROI 2048x4, Mono16 | 9746.6 | 9491.33 / 9368.06 / 9489.10 | 9449.50 | 97% |

## Inter-frame timing (consumer side)

| Case | Mean dt (ms) | p99 dt (ms) | Max dt (ms), runs 1/2/3 |
|---|---|---|---|
| Full, Mono16 | 22.30 | 25.7-28.4 | 29.29 / 26.42 / 28.55 |
| Full, Mono12Packed | 23.5-24.0 | 29.5-30.5 | 36.87 / 30.73 / 32.58 |
| 2048x256 | 2.77 | 5.4-5.7 | 6.22 / 6.56 / 6.52 |
| 2048x16 | 0.23 | 0.33 | 3.38 / 4.14 / 4.04 |
| 2048x4 | 0.11 | 0.19 | 0.46 / 0.94 / 0.40 |

## Overheads

| Operation | Time |
|---|---|
| `open()` (initialise + AT_Open + geometry) | 0.92-0.93 s |
| `get_float("ExposureTime")` | about 36 us |
| `set_float("ExposureTime")` | about 23.9 ms |
| `configure()` | 0.07-0.33 s depending on case |
| `prepare()` + `start()` | 0.09-0.48 s |
| `stop()` (AcquisitionStop + flush) | 5-19 ms |

## Observations

- Mono16 at full frame is 5-6% under the SDK limit; the 4-row ROI is about 3% under.
  Both are consumer-side cost (copy and requeue) or USB limits rather than wrapper overhead.
- Mono12Packed is the weak spot: 42 fps against the 47.5 fps limit. The cost is
  `SDK3BufferPool._mono12packed_array`, which unpacks in numpy into a fresh array
  for every frame. Mono16, Mono12 and Mono32 are zero-copy views and don't have this cost.
- At 4.4 kHz and above, the worst gap between frames stays within a few ms.
- Not measured: dropped-frame counts (frame counts were only compared to the SDK
  FrameRate), long-duration stability, and external-trigger mode.

## Reproduce

Stand-alone script (loads the wrapper by file path to avoid importing pymodaq):

```python
import sys, time, json, importlib.util
import numpy as np

SRC = r"C:\Users\ATTOExpFa1b\PycharmProjects\pymodaq_plugins_fab1b_5.2\src\pymodaq_plugins_fab1b\hardware\andor_sdk3\andor_sdk3.py"
DLL = r"C:\Program Files\Andor SOLIS\atcore.dll"
DUR = 5.0

# (label, roi (left, top, w, h) or None for full frame, encoding)
CASES = [
    ("full 2048x2048 Mono16", None, "Mono16"),
    ("full 2048x2048 Mono12Packed", None, "Mono12Packed"),
    ("ROI 2048x256 Mono16", (1, 900, 2048, 256), "Mono16"),
    ("ROI 2048x16 Mono16", (1, 1000, 2048, 16), "Mono16"),
    ("ROI 2048x4 Mono16", (1, 1000, 2048, 4), "Mono16"),
]

spec = importlib.util.spec_from_file_location("andor_sdk3", SRC)
m = importlib.util.module_from_spec(spec)
sys.modules["andor_sdk3"] = m
spec.loader.exec_module(m)

cam = m.AndorSDK3Camera(0, dll_path=DLL, n_buffers=32)
cam.open()
for label, roi, enc in CASES:
    cam.configure(exposure_s="min", roi=roi, pixel_encoding=enc, frame_rate="max")
    sdk_fps = cam.get_float("FrameRate")
    cam.prepare()
    cam.start()
    n, ts, t0 = 0, [], time.perf_counter()
    while time.perf_counter() - t0 < DUR:
        r = cam.wait_frame(1000)
        if r is None:
            continue
        buf, size = r
        img = cam.frame_view(buf).copy()
        cam.requeue(buf)
        n += 1
        ts.append(time.perf_counter())
    wall = time.perf_counter() - t0
    cam.stop()
    cam.buffer_pool = None
    dt = np.diff(ts) * 1e3
    print(f"{label}: {n / wall:.1f} fps (SDK {sdk_fps:.1f}), "
          f"dt mean {dt.mean():.3f} ms, p99 {np.percentile(dt, 99):.3f}, max {dt.max():.3f}")
cam.close()
```
