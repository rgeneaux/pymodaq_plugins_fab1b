pymodaq_plugins_greateyes (Greateyes)
#############################################

.. image:: https://img.shields.io/pypi/v/pymodaq_plugins_greateyes.svg
   :target: https://pypi.org/project/pymodaq_plugins_greateyes/
   :alt: Latest Version

.. image:: https://readthedocs.org/projects/pymodaq/badge/?version=latest
   :target: https://pymodaq.readthedocs.io/en/stable/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/CEMES-CNRS/pymodaq_plugins_greateyes/workflows/Upload%20Python%20Package/badge.svg
    :target: https://github.com/CEMES-CNRS/pymodaq_plugins_greateyes

PyMoDAQ plugin for instruments from Greateyes (ALEX, ELSE, GE XXXX)


Authors
=======

* Romain Geneaux

Instruments
===========
Below is the list of instruments included in this plugin

Viewer2D
+++++++++

* **GreateyesCCD**: Greateyes CCD cameras using the SDK

Installation notes
==================
This plugin uses the Greateyes C++ SDK (greateyes.dll) and the python wrapper developed by Greateyes. If you do not have either of them, contact Greateyes. 
Before using, you need to copy greateyesSDK.py, greateyes.dll and geCommLib.dll to the hardware folder of this plugin.

Before using the plugin, it is recommended to test your connection to the camera with the greateyes Vision software.

Notes: Options to choose "SensorOutputMode" in cameras with several output nodes (2048 px x 2048 px, or 4096px x 4096 px) is not implemented yet.
Tested on Windows10 with pymodaq 3.1.0, using USB connection.
