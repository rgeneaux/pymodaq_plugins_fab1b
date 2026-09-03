pymodaq_plugins_maranax
=======================

PyMoDAQ 5.2 plugin for Andor Marana-X cameras using the native Andor SDK3
``atcore`` API.

This project is intentionally independent of PyLabLib for acquisition. The
camera is configured through SDK3 features and images are acquired using the
native AT_QueueBuffer / AT_WaitBuffer interface.

Current status
--------------

* Native SDK3 ctypes wrapper.
* Native SDK3 circular buffer.
* Mono16 zero-copy NumPy view of the SDK buffer.
* One controlled copy from the SDK buffer into a PyMoDAQ-owned frame buffer.
* Dedicated acquisition thread.
* ROI and exposure control.
* Continuous/live acquisition.
* Basic frame-rate and dropped-frame accounting.

The current implementation deliberately uses Mono16 as the acquisition
encoding. Mono12Packed conversion can be added once the exact Marana-X
firmware/SDK combination has been tested.

Installation
------------

Install the package in editable mode in the same environment as PyMoDAQ::

    pip install -e .

Install the Andor SDK3 software supplied with the camera. The ``atcore.dll``
must be discoverable by the process, or its full path can be selected in the
PyMoDAQ settings.

The code has not been tested against every SDK3 release. Verify the installed
SDK3 version and the Marana-X firmware before using it for unattended
acquisition.

Performance design
------------------

The acquisition path is::

    Marana-X -> SDK3 -> preallocated native buffers -> AT_WaitBuffer
              -> NumPy view -> PyMoDAQ-owned frame -> dte_signal

The SDK buffer is returned immediately after the copy. It is therefore never
held by the GUI. The PyMoDAQ-facing frame has independent ownership.

For maximum acquisition throughput, keep processing out of the Qt slot
connected to ``frame_ready``. A future high-throughput recorder should consume
the acquisition frames in a separate producer/consumer queue.
