"""Serial driver for a Uniblitz VLM1 shutter driver (channel 1).

Write-only protocol at 9600 baud: ``b"@"`` opens the shutter and ``b"A"``
closes it. The controller never answers, so :attr:`VLM1Shutter.is_open` is the
last state commanded, not a measurement.

Imports neither Qt nor PyMoDAQ, so it can be used and tested stand-alone.
"""

from __future__ import annotations

import threading

import serial
from serial.tools import list_ports

OPEN_COMMAND = b"@"
CLOSE_COMMAND = b"A"
BAUDRATE = 9600
PREFERRED_PORT = "COM10"


def available_ports():
    return [port.device for port in list_ports.comports()]


def default_port(ports=None):
    """COM10 if present (where the shutter has always been), else the first port."""
    ports = available_ports() if ports is None else ports
    if PREFERRED_PORT in ports:
        return PREFERRED_PORT
    return ports[0] if ports else None


class VLM1Shutter:
    """One shutter on one serial port. Thread-safe: writes are serialized."""

    def __init__(self, port, baudrate=BAUDRATE, write_timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.write_timeout = write_timeout
        self._serial = None
        self._is_open = False
        self._lock = threading.Lock()

    @property
    def is_connected(self):
        return self._serial is not None and bool(getattr(self._serial, "is_open", True))

    @property
    def is_open(self):
        """Last commanded state (True = shutter open)."""
        return self._is_open

    def connect(self):
        if self.is_connected:
            return
        # Looked up at call time (not bound as a default argument) so tests can
        # substitute serial.Serial.
        self._serial = serial.Serial(
            self.port, baudrate=self.baudrate, write_timeout=self.write_timeout
        )

    def disconnect(self):
        with self._lock:
            serial_port, self._serial = self._serial, None
        if serial_port is not None:
            serial_port.close()

    def set_open(self, open_):
        with self._lock:
            if not self.is_connected:
                raise RuntimeError(f"VLM1 shutter on {self.port} is not connected")
            self._serial.write(OPEN_COMMAND if open_ else CLOSE_COMMAND)
            self._serial.flush()
            self._is_open = bool(open_)

    def open_shutter(self):
        self.set_open(True)

    def close_shutter(self):
        self.set_open(False)
