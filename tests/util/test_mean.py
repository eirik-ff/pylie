import numpy as np

from pylie import *
from pylie.util import *


def main():
    def test_SE3():
        # Define the pose distribution.
        mean_pose = SE3()
        cov_pose = np.diag(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.4]) ** 2)

        N = 100
        poses = [SE3() for _ in range(N)]
        poses_SE3_matrix = []

        # Draw random tangent space vectors.
        random_xis = np.random.multivariate_normal(np.zeros(6), cov_pose, N).T

        # Perturb the mean pose with each of the random tangent space vectors.
        for i in range(N):
            poses[i] = poses[i] + random_xis[0:, i]
            poses_SE3_matrix.append(poses[i].to_matrix())

        # Estimate the mean pose from the random poses.
        estimated_mean_pose = mean(poses)
        estimated_mean_pose_matrix = mean_SE3_matrix(poses_SE3_matrix)
        print("True mean:\n", mean_pose.to_matrix())
        print("Esimtated mean:\n", estimated_mean_pose.to_matrix())
        print("Esimtated mean from matrices:\n", estimated_mean_pose_matrix)

    test_SE3()


if __name__ == "__main__":
    main()