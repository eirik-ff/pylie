from .. import *
import numpy as np
from typing import List, Union

def lerp(X1: Union[SE3, SE2, SO3, SO2], 
         X2: Union[SE3, SE2, SO3, SO2], 
         alpha: float) -> Union[SE3, SE2, SO3, SO2]:
    """Perform linear interpolation on the manifold
    :param alpha: A scalar interpolation factor in [0, 1]
    :param X_1: First element
    :param X_2: Second element
    :return: The interpolated element
    """
    assert 0 <= alpha <= 1, "Alpha must be in range [0, 1]."
    return X1 + alpha * (X2 - X1)


def lerp_between(X1: Union[SE3, SE2, SO3, SO2], 
                 X2: Union[SE3, SE2, SO3, SO2], 
                 N: int = 100) -> List[Union[SE3, SE2, SO3, SO2]]:
    """Perform N linear interpolations on the manifold between X1 and X2. 
    :param alpha: A scalar interpolation factor in [0, 1]
    :param X_1: First element
    :param X_2: Second element
    :return: The interpolated element
    """
    lerpd = []
    for alpha in np.linspace(0, 1, N):
        lerpd.append(lerp(X1, X2, alpha))
    return lerpd
