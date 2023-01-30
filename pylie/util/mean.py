from .. import *
import numpy as np
import numpy.typing as npt
from typing import List, Union

DEFAULT_CONV_THRESH = 1e-14
DEFAULT_MAX_ITERS = 20


def mean(poses: List[Union[SE3, SE2, SO3, SO2]], 
         conv_thresh=DEFAULT_CONV_THRESH, 
         max_iters=DEFAULT_MAX_ITERS) -> Union[SE3, SE2, SO3, SO2]:
    """Calculate the mean pose from a list of poses using an iterative 
    algorithm.
    
    :param poses: List of poses
    :param conv_thresh: Convergence threshold for norm of tangent vector
    :param max_iters: Maximum iterations used in the algorithm
    
    :return: Mean pose of same type as input poses
    """
    N = len(poses)
    mean_pose = poses[0]

    for it in range(max_iters):
        # Compute the mean tangent vector in the tangent space at the current estimate.
        mean_xi = np.zeros((6,))
        for pose in poses:
            mean_xi = mean_xi + (pose - mean_pose)
        mean_xi = mean_xi / N

        # Update the estimate.
        mean_pose = mean_pose + mean_xi

        # Stop if the update is small.
        if np.linalg.norm(mean_xi) < conv_thresh:
            break

    return mean_pose


def mean_SE3_matrix(poses_SE3_matrix: List[npt.NDArray], 
                    conv_thresh=DEFAULT_CONV_THRESH, 
                    max_iters=DEFAULT_MAX_ITERS) -> npt.NDArray:
    """Wrapper function for `mean` where the SE3 poses are given as 4x4 
    matrices.
    
    :param poses_SE3_matrix: List of poses as 4x4 SE3 matrices
    :param conv_thresh: Convergence threshold for norm of tangent vector
    :param max_iters: Maximum iterations used in the algorithm
    
    :return: Mean pose as 4x4 SE3 matrix
    """                
    poses = []
    for T in poses_SE3_matrix:
        pose = SE3.from_matrix(T)
        poses.append(pose)

    return mean(poses, conv_thresh, max_iters).to_matrix()
