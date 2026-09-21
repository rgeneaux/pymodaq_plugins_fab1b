"""VLM1 shutter driver: protocol bytes and state, against a fake serial port."""
import unittest
from unittest import mock

from pymodaq_plugins_fab1b.hardware import uniblitz_vlm1 as vlm1


class FakeSerial:
    instances = []

    def __init__(self, port, baudrate=9600, write_timeout=None):
        self.port = port
        self.baudrate = baudrate
        self.write_timeout = write_timeout
        self.writes = []
        self.is_open = True
        FakeSerial.instances.append(self)

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        pass

    def close(self):
        self.is_open = False


class VLM1ShutterTest(unittest.TestCase):
    def setUp(self):
        FakeSerial.instances = []
        patcher = mock.patch.object(vlm1.serial, "Serial", FakeSerial)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_open_close_send_protocol_bytes(self):
        shutter = vlm1.VLM1Shutter("COM10")
        shutter.connect()
        shutter.open_shutter()
        shutter.close_shutter()
        port = FakeSerial.instances[0]
        self.assertEqual(port.writes, [b"@", b"A"])
        self.assertEqual((port.port, port.baudrate), ("COM10", 9600))

    def test_is_open_tracks_last_command(self):
        shutter = vlm1.VLM1Shutter("COM10")
        shutter.connect()
        self.assertFalse(shutter.is_open)
        shutter.set_open(True)
        self.assertTrue(shutter.is_open)
        shutter.set_open(False)
        self.assertFalse(shutter.is_open)

    def test_writing_while_disconnected_raises(self):
        shutter = vlm1.VLM1Shutter("COM10")
        with self.assertRaises(RuntimeError):
            shutter.open_shutter()
        shutter.connect()
        shutter.disconnect()
        with self.assertRaises(RuntimeError):
            shutter.close_shutter()

    def test_disconnect_closes_the_port(self):
        shutter = vlm1.VLM1Shutter("COM10")
        shutter.connect()
        shutter.disconnect()
        self.assertFalse(FakeSerial.instances[0].is_open)
        self.assertFalse(shutter.is_connected)

    def test_connect_is_idempotent(self):
        shutter = vlm1.VLM1Shutter("COM10")
        shutter.connect()
        shutter.connect()
        self.assertEqual(len(FakeSerial.instances), 1)


class DefaultPortTest(unittest.TestCase):
    def test_prefers_com10(self):
        self.assertEqual(vlm1.default_port(["COM1", "COM10", "COM5"]), "COM10")

    def test_falls_back_to_first_port(self):
        self.assertEqual(vlm1.default_port(["COM5", "COM7"]), "COM5")

    def test_none_without_ports(self):
        self.assertIsNone(vlm1.default_port([]))


if __name__ == "__main__":
    unittest.main()
