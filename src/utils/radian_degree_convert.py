import math
import numpy as np
from typing import Any


def radian2degree(radian: Any):
    return radian * 180 / math.pi


def degree2radian(degree: Any):
    return degree * math.pi / 180
