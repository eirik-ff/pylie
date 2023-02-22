from pylie import *
from pylie.util import *
import numpy as np


def main():
    def test_lerp():
        T1 = SE3(SO3.from_roll_pitch_yaw(np.pi / 4, 0, np.pi / 2), np.array([1, 1, 1]))
        T2 = SE3(SO3.from_roll_pitch_yaw(-np.pi / 6, np.pi / 4, np.pi / 2), np.array([1, 4, 2]))
        Ts = lerp_between(T1, T2, 5)
        for i, alpha in enumerate(np.linspace(0, 1, 5)):
            print(f"Lerp at alpha = {alpha}\n{Ts[i]}\n---")

    test_lerp()


if __name__ == "__main__":
    main()
