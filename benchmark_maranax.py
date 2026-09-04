"""Standalone SDK3 benchmark for an Andor Marana-X.

Run this outside PyMoDAQ first. It measures the camera/SDK3 acquisition path
without GUI overhead.

Example:
    python benchmark_maranax.py --roi 512 512 --duration 10 --buffers 32
"""

from __future__ import annotations

import argparse
import time

from pymodaq_plugins_fab1b.hardware.andor_sdk3 import AndorSDK3Camera


def benchmark(camera, duration):
    camera.prepare()
    camera.start()

    frames = 0
    t0 = time.perf_counter()
    deadline = t0 + duration

    try:
        while time.perf_counter() < deadline:
            result = camera.wait_frame(timeout_ms=1000)
            if result is None:
                continue

            buffer, _size = result
            try:
                # Do not copy here: this benchmark measures SDK3 buffer
                # delivery, not NumPy/PyMoDAQ processing.
                _ = camera.frame_view(buffer)
                frames += 1
            finally:
                camera.requeue(buffer)
    finally:
        camera.stop()

    elapsed = time.perf_counter() - t0
    fps = frames / elapsed

    print()
    print("Andor Marana-X SDK3 benchmark")
    print("--------------------------------")
    print(f"ROI:             {camera.width} x {camera.height}")
    print(f"Pixel encoding:  {camera.pixel_encoding}")
    print(f"ImageSizeBytes:  {camera.image_size_bytes}")
    print(f"AOIStride:       {camera.stride}")
    print(f"Buffers:         {camera.n_buffers}")
    print(f"ExposureTime:    {camera.get_float('ExposureTime') * 1000:.3f} ms")
    print(f"ReadoutTime:     {camera.get_float('ReadoutTime') * 1000:.3f} ms")
    print(f"FrameRate (set): {camera.get_float('FrameRate'):.3f} Hz")
    print(f"Elapsed:         {elapsed:.3f} s")
    print(f"Frames:          {frames}")
    print(f"Measured FPS:    {fps:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", default="C:\\Program Files\\Andor SOLIS\\atcore.dll")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--buffers", type=int, default=32)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--roi",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
    )
    parser.add_argument(
        "--exposure",
        default="min",
        help="Exposure time in seconds, or 'min' for the camera's minimum "
        "(default: min, for max-throughput benchmarking)",
    )
    parser.add_argument(
        "--frame-rate",
        default="max",
        help="Target FrameRate in Hz, or 'max' for the ceiling given the "
        "current AOI/exposure (default: max)",
    )
    args = parser.parse_args()

    exposure_s = args.exposure if args.exposure == "min" else float(args.exposure)
    frame_rate = args.frame_rate if args.frame_rate == "max" else float(args.frame_rate)

    camera = AndorSDK3Camera(
        index=args.camera,
        dll_path=args.dll,
        n_buffers=args.buffers,
    )
    camera.open()

    try:
        roi = None
        if args.roi:
            width, height = args.roi
            roi = (camera.left, camera.top, width, height)

        camera.configure(
            roi=roi,
            pixel_encoding="Mono16",
            exposure_s=exposure_s,
            frame_rate=frame_rate,
        )

        benchmark(camera, args.duration)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
