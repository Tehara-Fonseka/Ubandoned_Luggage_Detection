import collections
import sys
from typing import Optional

import numpy as np
from loguru import logger

""" types """

# Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, xmax, ymax] format is accepted
Box = np.ndarray

# Vector is of shape (1, N)
Vector = np.ndarray

# Track is meant as an output from the object tracker
Track = collections.namedtuple('Track', 'id box label maturity')


class Detection:
    # Detection is to be an input the the tracker
    def __init__(
            self,
            box: Box,
            score: Optional[float] = None,
            feature: Optional[Vector] = None,
            label: Optional[int] = None): #TT label = {'person':0, 'luggage':1}
        self.box = box
        self.score = score
        self.feature = feature
        self.label = label #TT

    def __repr__(self):
        fmt = "(box) %s,\t(score) %s,\t(feature) %s\n"
        return fmt % (str(self.box),
                      str(self.score) or 'none',
                      str(self.feature) or 'none')


""" utils """


def set_log_level(level: str) -> None:
    logger.remove()
    logger.add(sys.stdout, level=level)
