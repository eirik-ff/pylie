from pylie import *
from pylie.util import *

import numpy as np


def main():
    def test_random():
        mean = SE3()
        cov = np.eye(6)
        rng = np.random.default_rng(0)

        p = random_perturbation_normal(cov, 4, rng)
        print("Random perturbations with covariance 1 on all axes:\n", p)
        print()

        p = random_pose_normal(mean, cov, rng)
        print("Identity SE3 perturbed with covariance 1 on all axes:\n", p)

    test_random()


if __name__ == "__main__":
    main()
