from __future__ import annotations

import ctypes
import functools
import os
import platform
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _locked(method):
    """Serialize access to the SDK3 handle.

    Concurrently calling into atcore.dll from two threads on the same
    handle (e.g. a feature get/set from the GUI thread while an acquisition
    thread is blocked inside AT_WaitBuffer) is not safe: it was observed to
    crash the process outright rather than raise a Python exception. Every
    method that touches self.lib must go through this.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


AT_H = ctypes.c_int
AT_U8 = ctypes.c_ubyte
AT_BOOL = ctypes.c_int
AT_64 = ctypes.c_int64

AT_SUCCESS = 0
AT_ERR_TIMEDOUT = 13
AT_ERR_BUFFERFULL = 14
AT_INFINITE = 0xFFFFFFFF

# The system handle used to query global (non-camera) features such as
# "DeviceCount". There is no dedicated AT_GetDeviceCount() function in SDK3.
AT_HANDLE_SYSTEM = 1

AT_TRUE = 1
AT_FALSE = 0

# The host-side numpy dtype each PixelEncoding decodes to. Mono12 and
# Mono12Packed both end up as uint16 (Mono12 is already stored unpacked, one
# 12-bit value per 16-bit word; Mono12Packed is unpacked into the same
# container by SDK3BufferPool.frame_view).
PIXEL_ENCODING_DTYPES = {
    "Mono16": np.uint16,
    "Mono12": np.uint16,
    "Mono12Packed": np.uint16,
    "Mono32": np.uint32,
}

# Raw SDK3 wire size per pixel, i.e. what ImageSizeBytes/AOIStride actually
# reflect - distinct from PIXEL_ENCODING_DTYPES above, which is the *decoded*
# host-side size. Only Mono12Packed differs (3 bytes per 2 pixels).
PIXEL_ENCODING_RAW_BYTES_PER_PIXEL = {
    "Mono16": 2.0,
    "Mono12": 2.0,
    "Mono12Packed": 1.5,
    "Mono32": 4.0,
}


class SDK3Error(RuntimeError):
    """An error returned by the Andor SDK3 API."""

    def __init__(self, function: str, code: int):
        self.function = function
        self.code = int(code)
        super().__init__(f"{function} failed with SDK3 error code {code}")


def _wc(value: str):
    """Convert a Python string to the SDK3 wide-character argument."""
    return ctypes.c_wchar_p(value)


@dataclass
class SDK3Buffer:
    """One application-owned buffer registered with SDK3."""

    raw: ctypes.Array
    address: int
    size: int
    index: int


class SDK3Library:
    """Thin ctypes wrapper around the native Andor SDK3 atcore library."""

    def __init__(self, dll_path: Optional[str] = None):
        if dll_path is None:
            dll_path = "atcore.dll" if platform.system() == "Windows" else "libatcore.so.3"

        self.dll_path = dll_path
        # Guards every call into atcore.dll for this handle; see _locked.
        self._lock = threading.RLock()

        if platform.system() == "Windows":
            # atcore.dll dynamically loads sibling device-backend DLLs at
            # runtime (e.g. atusb_libusb.dll for USB cameras like the
            # Marana) via a plain LoadLibrary("name.dll") call, which is
            # resolved through the classic PATH-based search order. Without
            # its own directory on PATH those loads fail silently and the
            # camera is simply never enumerated: AT_InitialiseLibrary still
            # succeeds, but DeviceCount reads 0. os.add_dll_directory() does
            # NOT fix this (it only affects the safe-search set, which
            # excludes PATH), so PATH must be extended explicitly.
            dll_dir = os.path.dirname(os.path.abspath(dll_path))
            if dll_dir and dll_dir not in os.environ["PATH"].split(os.pathsep):
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
            self.lib = ctypes.WinDLL(dll_path)
        else:
            self.lib = ctypes.CDLL(dll_path)

        self._initialised = False
        self._configure_api()

    def _configure_api(self):
        lib = self.lib

        lib.AT_InitialiseLibrary.argtypes = []
        lib.AT_InitialiseLibrary.restype = ctypes.c_int
        lib.AT_FinaliseLibrary.argtypes = []
        lib.AT_FinaliseLibrary.restype = ctypes.c_int

        lib.AT_Open.argtypes = [ctypes.c_int, ctypes.POINTER(AT_H)]
        lib.AT_Open.restype = ctypes.c_int
        lib.AT_Close.argtypes = [AT_H]
        lib.AT_Close.restype = ctypes.c_int

        lib.AT_IsImplemented.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(AT_BOOL)
        ]
        lib.AT_IsImplemented.restype = ctypes.c_int
        lib.AT_IsReadable.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(AT_BOOL)
        ]
        lib.AT_IsReadable.restype = ctypes.c_int
        lib.AT_IsWritable.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(AT_BOOL)
        ]
        lib.AT_IsWritable.restype = ctypes.c_int

        lib.AT_GetInt.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(AT_64)
        ]
        lib.AT_GetInt.restype = ctypes.c_int
        lib.AT_SetInt.argtypes = [AT_H, ctypes.c_wchar_p, AT_64]
        lib.AT_SetInt.restype = ctypes.c_int
        lib.AT_GetIntMin.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(AT_64)
        ]
        lib.AT_GetIntMin.restype = ctypes.c_int
        lib.AT_GetIntMax.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(AT_64)
        ]
        lib.AT_GetIntMax.restype = ctypes.c_int

        lib.AT_GetFloat.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_double)
        ]
        lib.AT_GetFloat.restype = ctypes.c_int
        lib.AT_SetFloat.argtypes = [AT_H, ctypes.c_wchar_p, ctypes.c_double]
        lib.AT_SetFloat.restype = ctypes.c_int
        lib.AT_GetFloatMin.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_double)
        ]
        lib.AT_GetFloatMin.restype = ctypes.c_int
        lib.AT_GetFloatMax.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_double)
        ]
        lib.AT_GetFloatMax.restype = ctypes.c_int

        lib.AT_GetBool.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(AT_BOOL)
        ]
        lib.AT_GetBool.restype = ctypes.c_int
        lib.AT_SetBool.argtypes = [AT_H, ctypes.c_wchar_p, AT_BOOL]
        lib.AT_SetBool.restype = ctypes.c_int

        lib.AT_GetEnumCount.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)
        ]
        lib.AT_GetEnumCount.restype = ctypes.c_int
        lib.AT_GetEnumStringByIndex.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.c_int,
            ctypes.c_wchar_p, ctypes.c_int
        ]
        lib.AT_GetEnumStringByIndex.restype = ctypes.c_int
        lib.AT_IsEnumIndexAvailable.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(AT_BOOL)
        ]
        lib.AT_IsEnumIndexAvailable.restype = ctypes.c_int
        lib.AT_IsEnumIndexImplemented.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(AT_BOOL)
        ]
        lib.AT_IsEnumIndexImplemented.restype = ctypes.c_int
        lib.AT_GetEnumIndex.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)
        ]
        lib.AT_GetEnumIndex.restype = ctypes.c_int
        lib.AT_SetEnumIndex.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.c_int
        ]
        lib.AT_SetEnumIndex.restype = ctypes.c_int
        lib.AT_SetEnumString.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.c_wchar_p
        ]
        lib.AT_SetEnumString.restype = ctypes.c_int

        lib.AT_GetStringMaxLength.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)
        ]
        lib.AT_GetStringMaxLength.restype = ctypes.c_int
        lib.AT_GetString.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int
        ]
        lib.AT_GetString.restype = ctypes.c_int
        lib.AT_SetString.argtypes = [
            AT_H, ctypes.c_wchar_p, ctypes.c_wchar_p
        ]
        lib.AT_SetString.restype = ctypes.c_int

        lib.AT_Command.argtypes = [AT_H, ctypes.c_wchar_p]
        lib.AT_Command.restype = ctypes.c_int

        lib.AT_QueueBuffer.argtypes = [
            AT_H, ctypes.POINTER(AT_U8), ctypes.c_int
        ]
        lib.AT_QueueBuffer.restype = ctypes.c_int
        lib.AT_WaitBuffer.argtypes = [
            AT_H, ctypes.POINTER(ctypes.POINTER(AT_U8)),
            ctypes.POINTER(ctypes.c_int), ctypes.c_uint
        ]
        lib.AT_WaitBuffer.restype = ctypes.c_int
        lib.AT_Flush.argtypes = [AT_H]
        lib.AT_Flush.restype = ctypes.c_int

    @staticmethod
    def check(code, function):
        if code != AT_SUCCESS:
            raise SDK3Error(function, code)

    @_locked
    def initialise(self):
        if not self._initialised:
            self.check(self.lib.AT_InitialiseLibrary(), "AT_InitialiseLibrary")
            self._initialised = True

    @_locked
    def finalise(self):
        if self._initialised:
            self.check(self.lib.AT_FinaliseLibrary(), "AT_FinaliseLibrary")
            self._initialised = False

    def device_count(self):
        # SDK3 has no AT_GetDeviceCount(); device count is the "DeviceCount"
        # integer feature on the system handle.
        return self.get_int(AT_HANDLE_SYSTEM, "DeviceCount")

    @_locked
    def open(self, index=0):
        handle = AT_H()
        self.check(
            self.lib.AT_Open(index, ctypes.byref(handle)),
            "AT_Open",
        )
        return handle

    @_locked
    def close(self, handle):
        self.check(self.lib.AT_Close(handle), "AT_Close")

    @_locked
    def is_implemented(self, handle, feature):
        value = AT_BOOL()
        self.check(
            self.lib.AT_IsImplemented(handle, _wc(feature), ctypes.byref(value)),
            f"AT_IsImplemented({feature})",
        )
        return bool(value.value)

    @_locked
    def is_readable(self, handle, feature):
        value = AT_BOOL()
        self.check(
            self.lib.AT_IsReadable(handle, _wc(feature), ctypes.byref(value)),
            f"AT_IsReadable({feature})",
        )
        return bool(value.value)

    @_locked
    def is_writable(self, handle, feature):
        value = AT_BOOL()
        self.check(
            self.lib.AT_IsWritable(handle, _wc(feature), ctypes.byref(value)),
            f"AT_IsWritable({feature})",
        )
        return bool(value.value)

    @_locked
    def get_int(self, handle, feature):
        value = AT_64()
        self.check(
            self.lib.AT_GetInt(handle, _wc(feature), ctypes.byref(value)),
            f"AT_GetInt({feature})",
        )
        return int(value.value)

    @_locked
    def set_int(self, handle, feature, value):
        self.check(
            self.lib.AT_SetInt(handle, _wc(feature), int(value)),
            f"AT_SetInt({feature})",
        )

    @_locked
    def get_int_min(self, handle, feature):
        value = AT_64()
        self.check(
            self.lib.AT_GetIntMin(handle, _wc(feature), ctypes.byref(value)),
            f"AT_GetIntMin({feature})",
        )
        return int(value.value)

    @_locked
    def get_int_max(self, handle, feature):
        value = AT_64()
        self.check(
            self.lib.AT_GetIntMax(handle, _wc(feature), ctypes.byref(value)),
            f"AT_GetIntMax({feature})",
        )
        return int(value.value)

    @_locked
    def get_float(self, handle, feature):
        value = ctypes.c_double()
        self.check(
            self.lib.AT_GetFloat(handle, _wc(feature), ctypes.byref(value)),
            f"AT_GetFloat({feature})",
        )
        return float(value.value)

    @_locked
    def set_float(self, handle, feature, value):
        self.check(
            self.lib.AT_SetFloat(handle, _wc(feature), float(value)),
            f"AT_SetFloat({feature})",
        )

    @_locked
    def get_float_min(self, handle, feature):
        value = ctypes.c_double()
        self.check(
            self.lib.AT_GetFloatMin(handle, _wc(feature), ctypes.byref(value)),
            f"AT_GetFloatMin({feature})",
        )
        return float(value.value)

    @_locked
    def get_float_max(self, handle, feature):
        value = ctypes.c_double()
        self.check(
            self.lib.AT_GetFloatMax(handle, _wc(feature), ctypes.byref(value)),
            f"AT_GetFloatMax({feature})",
        )
        return float(value.value)

    @_locked
    def get_bool(self, handle, feature):
        value = AT_BOOL()
        self.check(
            self.lib.AT_GetBool(handle, _wc(feature), ctypes.byref(value)),
            f"AT_GetBool({feature})",
        )
        return bool(value.value)

    @_locked
    def set_bool(self, handle, feature, value):
        self.check(
            self.lib.AT_SetBool(handle, _wc(feature), AT_TRUE if value else AT_FALSE),
            f"AT_SetBool({feature})",
        )

    @_locked
    def get_enum_values(self, handle, feature, available_only=True):
        count = ctypes.c_int()
        self.check(
            self.lib.AT_GetEnumCount(handle, _wc(feature), ctypes.byref(count)),
            f"AT_GetEnumCount({feature})",
        )
        values = []
        for index in range(count.value):
            implemented = AT_BOOL()
            available = AT_BOOL()
            self.check(
                self.lib.AT_IsEnumIndexImplemented(
                    handle, _wc(feature), index, ctypes.byref(implemented)
                ),
                f"AT_IsEnumIndexImplemented({feature})",
            )
            if not implemented.value:
                continue
            self.check(
                self.lib.AT_IsEnumIndexAvailable(
                    handle, _wc(feature), index, ctypes.byref(available)
                ),
                f"AT_IsEnumIndexAvailable({feature})",
            )
            if available_only and not available.value:
                continue

            # 256 wchar characters is ample for normal SDK3 enum values.
            text = ctypes.create_unicode_buffer(256)
            self.check(
                self.lib.AT_GetEnumStringByIndex(
                    handle, _wc(feature), index, text, len(text)
                ),
                f"AT_GetEnumStringByIndex({feature})",
            )
            values.append(text.value)
        return values

    @_locked
    def get_enum(self, handle, feature):
        index = ctypes.c_int()
        self.check(
            self.lib.AT_GetEnumIndex(handle, _wc(feature), ctypes.byref(index)),
            f"AT_GetEnumIndex({feature})",
        )
        text = ctypes.create_unicode_buffer(256)
        self.check(
            self.lib.AT_GetEnumStringByIndex(
                handle, _wc(feature), index.value, text, len(text)
            ),
            f"AT_GetEnumStringByIndex({feature})",
        )
        return text.value

    @_locked
    def set_enum(self, handle, feature, value):
        self.check(
            self.lib.AT_SetEnumString(handle, _wc(feature), _wc(value)),
            f"AT_SetEnumString({feature})",
        )

    @_locked
    def get_string(self, handle, feature):
        max_length = ctypes.c_int()
        self.check(
            self.lib.AT_GetStringMaxLength(
                handle, _wc(feature), ctypes.byref(max_length)
            ),
            f"AT_GetStringMaxLength({feature})",
        )
        text = ctypes.create_unicode_buffer(max_length.value + 1)
        self.check(
            self.lib.AT_GetString(
                handle, _wc(feature), text, len(text)
            ),
            f"AT_GetString({feature})",
        )
        return text.value

    @_locked
    def set_string(self, handle, feature, value):
        self.check(
            self.lib.AT_SetString(handle, _wc(feature), _wc(value)),
            f"AT_SetString({feature})",
        )

    @_locked
    def command(self, handle, feature):
        self.check(
            self.lib.AT_Command(handle, _wc(feature)),
            f"AT_Command({feature})",
        )

    @_locked
    def queue_buffer(self, handle, address, size):
        ptr = ctypes.cast(address, ctypes.POINTER(AT_U8))
        self.check(
            self.lib.AT_QueueBuffer(handle, ptr, int(size)),
            "AT_QueueBuffer",
        )

    @_locked
    def wait_buffer(self, handle, timeout_ms=1000):
        pointer = ctypes.POINTER(AT_U8)()
        size = ctypes.c_int()
        code = self.lib.AT_WaitBuffer(
            handle,
            ctypes.byref(pointer),
            ctypes.byref(size),
            int(timeout_ms),
        )
        if code == AT_ERR_TIMEDOUT:
            return None, 0
        self.check(code, "AT_WaitBuffer")
        return ctypes.addressof(pointer.contents), size.value

    @_locked
    def flush(self, handle):
        self.check(self.lib.AT_Flush(handle), "AT_Flush")


class SDK3BufferPool:
    """Preallocated, SDK3-owned application buffer pool."""

    def __init__(self, sdk, handle, image_size_bytes, n_buffers=32):
        self.sdk = sdk
        self.handle = handle
        self.image_size_bytes = int(image_size_bytes)
        self.n_buffers = int(n_buffers)
        self.buffers = []
        self._by_address = {}
        self._allocate()

    def _allocate(self):
        for index in range(self.n_buffers):
            # SDK3 requires an 8-byte-aligned address.
            raw = ctypes.create_string_buffer(self.image_size_bytes + 7)
            base = ctypes.addressof(raw)
            address = (base + 7) & ~7
            buf = SDK3Buffer(
                raw=raw,
                address=address,
                size=self.image_size_bytes,
                index=index,
            )
            self.buffers.append(buf)
            self._by_address[address] = buf

    def queue_all(self):
        for buf in self.buffers:
            self.sdk.queue_buffer(self.handle, buf.address, buf.size)

    def find(self, address):
        try:
            return self._by_address[address]
        except KeyError as exc:
            raise RuntimeError(
                f"SDK3 returned an unknown buffer address 0x{address:x}"
            ) from exc

    @staticmethod
    def frame_view(buf, width, height, stride, pixel_encoding):
        """Return a view/array of the SDK buffer for the given PixelEncoding.

        Respects AOIStride padding. Mono16/Mono12/Mono32 are "dense": a fixed
        number of bytes per pixel, so this returns a real *view* into the
        SDK3 buffer with no copy. Mono12Packed is bit-packed (3 bytes per 2
        pixels) and must be unpacked into a new array; there is no way to
        alias that as a view.
        """
        if pixel_encoding in SDK3BufferPool._DENSE_DTYPES:
            return SDK3BufferPool._dense_view(
                buf, width, height, stride, SDK3BufferPool._DENSE_DTYPES[pixel_encoding]
            )
        if pixel_encoding == "Mono12Packed":
            return SDK3BufferPool._mono12packed_array(buf, width, height, stride)
        raise RuntimeError(f"PixelEncoding {pixel_encoding!r} is not supported")

    _DENSE_DTYPES = {"Mono16": np.uint16, "Mono12": np.uint16, "Mono32": np.uint32}

    @staticmethod
    def _dense_view(buf, width, height, stride, dtype):
        itemsize = np.dtype(dtype).itemsize
        if stride % itemsize:
            raise RuntimeError(
                f"AOIStride={stride} is not divisible by the {itemsize}-byte pixel size"
            )

        n_items_per_row = stride // itemsize
        raw = (ctypes.c_ubyte * buf.size).from_address(buf.address)
        flat = np.ctypeslib.as_array(raw).view(dtype)

        if flat.size < n_items_per_row * height:
            raise RuntimeError(
                "SDK3 ImageSizeBytes is smaller than AOIStride*AOIHeight"
            )

        image_with_padding = flat[:n_items_per_row * height].reshape(
            height, n_items_per_row
        )
        return image_with_padding[:, :width]

    @staticmethod
    def _mono12packed_array(buf, width, height, stride):
        # Andor's Mono12Packed layout, 2 pixels per 3 bytes (little-endian):
        #   byte0 = px0[7:0]
        #   byte1 = px1[3:0] << 4 | px0[11:8]
        #   byte2 = px1[11:4]
        raw = (ctypes.c_ubyte * buf.size).from_address(buf.address)
        flat = np.ctypeslib.as_array(raw)

        if flat.size < stride * height:
            raise RuntimeError(
                "SDK3 ImageSizeBytes is smaller than AOIStride*AOIHeight"
            )

        rows = flat[:stride * height].reshape(height, stride).astype(np.uint16)
        fst, mid, lst = rows[:, 0::3], rows[:, 1::3], rows[:, 2::3]
        n_pairs = min(mid.shape[1], lst.shape[1])

        unpacked = np.empty((height, 2 * n_pairs), dtype=np.uint16)
        unpacked[:, 0::2] = (fst[:, :n_pairs] << 4) | (mid[:, :n_pairs] & 0x0F)
        unpacked[:, 1::2] = (mid[:, :n_pairs] >> 4) | (lst[:, :n_pairs] << 4)
        return unpacked[:, :width]


class SDK3UtilityLibrary:
    """Thin ctypes wrapper around atutility.dll.

    Distinct from atcore.dll: this is where SDK3 puts its metadata-parsing
    helpers (AT_GetTimeStampFromMetadata etc.), rather than exposing them on
    the core library. Always installed alongside atcore.dll.
    """

    def __init__(self, dll_path):
        self.dll_path = dll_path
        self._lock = threading.RLock()
        self.lib = ctypes.WinDLL(dll_path) if platform.system() == "Windows" else ctypes.CDLL(dll_path)
        self._initialised = False
        self._configure_api()

    def _configure_api(self):
        lib = self.lib
        lib.AT_InitialiseUtilityLibrary.argtypes = []
        lib.AT_InitialiseUtilityLibrary.restype = ctypes.c_int
        lib.AT_FinaliseUtilityLibrary.argtypes = []
        lib.AT_FinaliseUtilityLibrary.restype = ctypes.c_int

        lib.AT_GetTimeStampFromMetadata.argtypes = [
            ctypes.POINTER(AT_U8), AT_64, ctypes.POINTER(AT_64)
        ]
        lib.AT_GetTimeStampFromMetadata.restype = ctypes.c_int

    @_locked
    def initialise(self):
        if not self._initialised:
            SDK3Library.check(self.lib.AT_InitialiseUtilityLibrary(), "AT_InitialiseUtilityLibrary")
            self._initialised = True

    @_locked
    def finalise(self):
        if self._initialised:
            SDK3Library.check(self.lib.AT_FinaliseUtilityLibrary(), "AT_FinaliseUtilityLibrary")
            self._initialised = False

    @_locked
    def get_timestamp_from_metadata(self, address, size):
        """Extract the per-frame hardware timestamp (in TimestampClock ticks).

        ``address``/``size`` are the buffer address and *total* size
        (image + metadata trailer) as returned by AT_WaitBuffer - the same
        pair SDK3Library.wait_buffer() already returns, not the plain
        ImageSizeBytes.
        """
        ptr = ctypes.cast(address, ctypes.POINTER(AT_U8))
        timestamp = AT_64()
        SDK3Library.check(
            self.lib.AT_GetTimeStampFromMetadata(ptr, AT_64(size), ctypes.byref(timestamp)),
            "AT_GetTimeStampFromMetadata",
        )
        return int(timestamp.value)


class AndorSDK3Camera:
    """Marana-X-oriented camera wrapper using native SDK3."""

    def __init__(self, index=0, dll_path=None, n_buffers=32):
        self.index = index
        self.sdk = SDK3Library(dll_path)
        self.utility = SDK3UtilityLibrary(self._sibling_dll_path(self.sdk.dll_path, "atutility.dll"))
        self.handle = None
        self.n_buffers = int(n_buffers)
        self.buffer_pool = None
        self.acquiring = False

        self.sensor_width = None
        self.sensor_height = None
        self.width = None
        self.height = None
        self.left = None
        self.top = None
        self.stride = None
        self.image_size_bytes = None
        self.pixel_encoding = None
        self.metadata_enabled = False

    @staticmethod
    def _sibling_dll_path(dll_path, sibling_name):
        """atutility.dll is always installed next to atcore.dll."""
        directory = os.path.dirname(os.path.abspath(dll_path))
        return os.path.join(directory, sibling_name) if directory else sibling_name

    def open(self):
        self.sdk.initialise()
        self.utility.initialise()
        try:
            self.handle = self.sdk.open(self.index)
            self._read_geometry()
        except Exception:
            self.sdk.finalise()
            self.utility.finalise()
            raise

    def close(self):
        if self.handle is not None:
            try:
                if self.acquiring:
                    self.stop()
            finally:
                self.sdk.close(self.handle)
                self.handle = None
        self.sdk.finalise()
        self.utility.finalise()

    def _read_geometry(self):
        self.sensor_width = self.sdk.get_int(self.handle, "SensorWidth")
        self.sensor_height = self.sdk.get_int(self.handle, "SensorHeight")
        self.width = self.sdk.get_int(self.handle, "AOIWidth")
        self.height = self.sdk.get_int(self.handle, "AOIHeight")
        self.left = self.sdk.get_int(self.handle, "AOILeft")
        self.top = self.sdk.get_int(self.handle, "AOITop")
        self.stride = self.sdk.get_int(self.handle, "AOIStride")
        self.image_size_bytes = self.sdk.get_int(
            self.handle, "ImageSizeBytes"
        )
        self.pixel_encoding = self.sdk.get_enum(
            self.handle, "PixelEncoding"
        )

    def get_int(self, feature):
        return self.sdk.get_int(self.handle, feature)

    def set_int(self, feature, value):
        self.sdk.set_int(self.handle, feature, value)

    def get_float(self, feature):
        return self.sdk.get_float(self.handle, feature)

    def set_float(self, feature, value):
        self.sdk.set_float(self.handle, feature, value)

    def get_bool(self, feature):
        return self.sdk.get_bool(self.handle, feature)

    def set_bool(self, feature, value):
        self.sdk.set_bool(self.handle, feature, value)

    def get_enum(self, feature):
        return self.sdk.get_enum(self.handle, feature)

    def set_enum(self, feature, value):
        self.sdk.set_enum(self.handle, feature, value)

    def enum_values(self, feature, available_only=True):
        return self.sdk.get_enum_values(
            self.handle, feature, available_only
        )

    def get_string(self, feature):
        return self.sdk.get_string(self.handle, feature)

    def is_implemented(self, feature):
        return self.sdk.is_implemented(self.handle, feature)

    def is_writable(self, feature):
        return self.sdk.is_writable(self.handle, feature)

    def configure(
        self,
        exposure_s=None,
        roi=None,
        pixel_encoding="Mono16",
        cycle_mode="Continuous",
        frame_rate=None,
        enable_metadata=False,
    ):
        if self.acquiring:
            raise RuntimeError("Cannot configure while acquiring")

        if pixel_encoding not in PIXEL_ENCODING_DTYPES:
            raise RuntimeError(
                f"Unsupported PixelEncoding {pixel_encoding!r}; supported: "
                f"{sorted(PIXEL_ENCODING_DTYPES)}"
            )
        self.set_enum("PixelEncoding", pixel_encoding)
        self.set_enum("CycleMode", cycle_mode)

        # Must happen before the final _read_geometry() below: ImageSizeBytes
        # only grows to include the metadata trailer (e.g. the per-frame
        # hardware timestamp) once MetadataEnable is set, and the buffer
        # pool is sized from that value - get this backwards and every
        # buffer is too small to hold the metadata SDK3 tries to append.
        self.set_bool("MetadataEnable", bool(enable_metadata))
        self.metadata_enabled = bool(enable_metadata)

        if roi is not None:
            left, top, width, height = map(int, roi)

            # Set dimensions before offsets; SDK3 cameras may constrain
            # offsets based on the selected AOI dimensions.
            self.set_int("AOIWidth", width)
            self.set_int("AOIHeight", height)
            self.set_int("AOILeft", left)
            self.set_int("AOITop", top)

        # ExposureTime/FrameRate are mutually constrained, and their bounds
        # depend on the AOI, so both must be set after the ROI above. Setting
        # exposure before FrameRate matters too: FrameRateMax is bounded by
        # the *current* exposure, so a stale long exposure silently caps how
        # high FrameRate can be pushed.
        if exposure_s is not None:
            if exposure_s == "min":
                exposure_s = self.sdk.get_float_min(self.handle, "ExposureTime")
            self.set_float("ExposureTime", exposure_s)

        if frame_rate is not None:
            if frame_rate == "max":
                frame_rate = self.sdk.get_float_max(self.handle, "FrameRate")
            self.set_float("FrameRate", frame_rate)

        self._read_geometry()

        self.buffer_pool = SDK3BufferPool(
            self.sdk,
            self.handle,
            self.image_size_bytes,
            self.n_buffers,
        )

    def prepare(self):
        if self.buffer_pool is None:
            self._read_geometry()
            self.buffer_pool = SDK3BufferPool(
                self.sdk,
                self.handle,
                self.image_size_bytes,
                self.n_buffers,
            )
        self.buffer_pool.queue_all()

    def start(self):
        if self.acquiring:
            return
        if self.buffer_pool is None:
            self.prepare()

        self.sdk.command(self.handle, "AcquisitionStart")
        self.acquiring = True

    def wait_frame(self, timeout_ms=1000):
        if not self.acquiring:
            raise RuntimeError("Acquisition is not running")

        address, size = self.sdk.wait_buffer(
            self.handle, timeout_ms
        )
        if address is None:
            return None

        return self.buffer_pool.find(address), size

    def frame_view(self, buffer):
        return self.buffer_pool.frame_view(
            buffer,
            self.width,
            self.height,
            self.stride,
            self.pixel_encoding,
        )

    @property
    def pixel_dtype(self):
        """The host-side numpy dtype frame_view() decodes to."""
        return PIXEL_ENCODING_DTYPES[self.pixel_encoding]

    def frame_timestamp(self, buffer, size):
        """Per-frame hardware timestamp, in TimestampClock ticks.

        Requires configure(enable_metadata=True). ``size`` must be the
        *total* size wait_frame() returned alongside this buffer (image +
        metadata trailer), not plain ImageSizeBytes.
        """
        if not self.metadata_enabled:
            raise RuntimeError("frame_timestamp() requires configure(enable_metadata=True)")
        return self.utility.get_timestamp_from_metadata(buffer.address, size)

    def requeue(self, buffer):
        self.sdk.queue_buffer(
            self.handle,
            buffer.address,
            buffer.size,
        )

    def stop(self):
        if not self.acquiring:
            return

        try:
            self.sdk.command(
                self.handle, "AcquisitionStop"
            )
        finally:
            # Required by SDK3 after AcquisitionStop.
            self.sdk.flush(self.handle)
            self.acquiring = False

    def device_info(self):
        result = {}
        for feature in (
            "CameraModel",
            "SerialNumber",
            "SoftwareVersion",
            "FirmwareVersion",
        ):
            try:
                if self.is_implemented(feature):
                    result[feature] = self.get_string(feature)
            except Exception:
                pass
        return result
