from .. import *
import numpy as np
import numpy.typing as npt
from typing import List, Union


def random_perturbation_normal(
        cov: npt.NDArray, 
        N: int = 1, 
        rng: np.random.Generator = None) -> npt.NDArray:
    """Draw N i.i.d. random poses with from a normal distribution with 
    covariance cov. 

    :param cov: Covariance of normal distribution, shape: tangent_dim x tangent_dim.
    :param N: Number of poses to draw.
    :param rng: Numpy random Generator. If None, default_rng is used.
    :return: A numpy array (shape: N x tangent_dim) of random perturbations.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    tangent_dim = cov.shape[0]
    perturb = rng.multivariate_normal(np.zeros(tangent_dim), cov, N)
    return perturb


def random_pose_normal(
        mean: Union[SE3, SE2, SO3, SO2], 
        cov: npt.NDArray, 
        rng: np.random.Generator = None) -> Union[SE3, SE2, SO3, SO2]:
    """Perturb a single pose with perturbation from a normal distribution with
    covariance cov. 

    :param mean: Pose to perturb.
    :param cov: Covariance matrix for the perturbation.
    :param rng: Numpy random Generator. If None, default_rng is used.
    :return: The perturbed pose
    """
    perturb = random_perturbation_normal(cov, 1, rng).squeeze()
    return mean + perturb
