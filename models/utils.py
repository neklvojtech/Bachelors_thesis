# =============================================================================
# File: utils.py
# Original Author: Vojtěch Vančura
# Description: Defines NMSE and get_first_item functions for the Beeformer model.
# Notes: Imported as part of the project without any modifications.
# =============================================================================


import os

os.environ["KERAS_BACKEND"] = "torch"

import keras

from _datasets.utils import *


def NMSE(x, y):
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    return keras.losses.mean_squared_error(x, y)

def get_first_item(d):
    return d[next(iter(d.keys()))]