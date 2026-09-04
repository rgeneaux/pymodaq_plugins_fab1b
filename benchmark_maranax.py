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
    """Run one acquisition and return the measured numbers as a dict.

    Does not copy frame data: this measures SDK3 buffer delivery, not
    NumPy/PyMoDAQ processing. frame_view() decoding cost (bit-unpacking for
    Mono12Packed) *is* included, since it happens before requeue() either
    way in real usage.
    """
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
                _ = camera.frame_view(buffer)
                frames += 1
            finally:
                camera.requeue(buffer)
    finally:
        camera.stop()

    elapsed = time.perf_counter() - t0

    return {
        "pixel_encoding": camera.pixel_encoding,
        "width": camera.width,
        "height": camera.height,
        "image_size_bytes": camera.image_size_bytes,
        "stride": camera.stride,
        "n_buffers": camera.n_buffers,
        "exposure_ms": camera.get_float("ExposureTime") * 1000,
        "readout_ms": camera.get_float("ReadoutTime") * 1000,
        "frame_rate_set_hz": camera.get_float("FrameRate"),
        "elapsed_s": elapsed,
        "frames": frames,
        "measured_fps": frames / elapsed,
    }


def print_result(result):
    print()
    print("Andor Marana-X SDK3 benchmark")
    print("--------------------------------")
    print(f"ROI:             {result['width']} x {result['height']}")
    print(f"Pixel encoding:  {result['pixel_encoding']}")
    print(f"ImageSizeBytes:  {result['image_size_bytes']}")
    print(f"AOIStride:       {result['stride']}")
    print(f"Buffers:         {result['n_buffers']}")
    print(f"ExposureTime:    {result['exposure_ms']:.3f} ms")
    print(f"ReadoutTime:     {result['readout_ms']:.3f} ms")
    print(f"FrameRate (set): {result['frame_rate_set_hz']:.3f} Hz")
    print(f"Elapsed:         {result['elapsed_s']:.3f} s")
    print(f"Frames:          {result['frames']}")
    print(f"Measured FPS:    {result['measured_fps']:.3f}")


def print_comparison(results):
    print()
    print("Andor Marana-X SDK3 benchmark - pixel encoding comparison")
    print(f"ROI: {results[0]['width']} x {results[0]['height']}")
    print("-" * 78)
    header = f"{'Encoding':<14}{'Bytes/frame':>13}{'Readout (ms)':>14}{'Measured FPS':>15}{'vs Mono16':>12}"
    print(header)
    print("-" * 78)
    baseline = next((r["measured_fps"] for r in results if r["pixel_encoding"] == "Mono16"), None)
    for r in results:
        ratio = f"{r['measured_fps'] / baseline:.2f}x" if baseline else "-"
        print(
            f"{r['pixel_encoding']:<14}{r['image_size_bytes']:>13}"
            f"{r['readout_ms']:>14.3f}{r['measured_fps']:>15.2f}{ratio:>12}"
        )


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
    parser.add_argument(
        "--pixel-encoding",
        default="Mono16",
        help="PixelEncoding to benchmark, or 'all' to run every encoding "
        "the camera supports in turn and print a comparison table "
        "(default: Mono16)",
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

        if args.pixel_encoding == "all":
            encodings = camera.enum_values("PixelEncoding")
            results = []
            for encoding in encodings:
                camera.configure(
                    roi=roi,
                    pixel_encoding=encoding,
                    exposure_s=exposure_s,
                    frame_rate=frame_rate,
                )
                results.append(benchmark(camera, args.duration))
                print_result(results[-1])
            print_comparison(results)
        else:
            camera.configure(
                roi=roi,
                pixel_encoding=args.pixel_encoding,
                exposure_s=exposure_s,
                frame_rate=frame_rate,
            )
            print_result(benchmark(camera, args.duration))
    finally:
        camera.close()


if __name__ == "__main__":
    main()
