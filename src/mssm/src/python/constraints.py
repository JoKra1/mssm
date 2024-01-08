import numpy as np
from dataclasses import dataclass
from enum import Enum


class ConstType(Enum):
    DROP = 1
    QR = 2

@dataclass
class Constraint:
  # Constraint storage. Either holds the Qr-based correction term that needs
  # to be applied to X, S, and D to make terms subject to sum-to-zero constraints (Wood, 2017)
  # or identifies the column/row that should be dropped from those - then X can also no longer take
  # on a constant.
  Z: np.array or int = None
  type:ConstType = None