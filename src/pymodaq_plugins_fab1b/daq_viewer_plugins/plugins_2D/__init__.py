import importlib
from pathlib import Path
from ...utils import set_logger
from ... import gui_patches
logger = set_logger('viewer2D_plugins', add_to_console=False)

gui_patches.apply()

for path in Path(__file__).parent.iterdir():
    try:
        if '__init__' not in str(path):
            importlib.import_module('.' + path.stem, __package__)
    except Exception as e:
        logger.warning("{:} plugin couldn't be loaded due to some missing packages or errors: {:}".format(path.stem, str(e)))
        pass


