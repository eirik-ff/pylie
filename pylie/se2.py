# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations
import numpy.typing as npt
from typing import Tuple

import numpy as np
from pylie import SO2


class SE2:
    """Represents an element of the SE(2) Lie group (poses in 2D)."""
    tangent_dim: int = 3

    def __init__(self, rotation: SO2 = SO2(), translation: npt.NDArray = np.zeros((2,))):
        """Constructs an SE(2) element.
        The default is the identity element.

        :param pose_tuple: A tuple (rotation (SO2), translation (2D column vector) (optional).
        """
        self.rotation: SO2 = rotation
        self.translation: npt.NDArray = translation

    @classmethod
    def from_matrix(cls, T: npt.NDArray) -> SE2:
        """Construct an SE(2) element corresponding from a pose matrix.
        The rotation is fitted to the closest rotation matrix, the bottom row of the 3x3 matrix is ignored.

        :param T: 3x3 or 2x3 pose matrix.
        :return: The SE(2) element.
        """
        return cls(SO2.from_matrix(T[:2, :2]), T[:2, 2])

    @property
    def rotation(self) -> SO2:
        """ The rotation, an element of SO(2)

        :return: An SO2 object corresponding to the orientation.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, so2: SO2):
        """Sets the rotation

        :param so2: An SO2
        """
        if not isinstance(so2, SO2):
            raise TypeError('Rotation must be a SO2')

        self._rotation = so2

    @property
    def translation(self) -> npt.NDArray:
        """The translation, a 2D column vector

        :return: A 2D column vector corresponding to the translation.
        """
        return self._translation

    @translation.setter
    def translation(self, t: npt.NDArray):
        """Sets the translation

        :param t: 2D column vector
        """
        if not isinstance(t, np.ndarray) and t.shape == (2,):
            raise TypeError('Translation must be a 2D column vector')

        # Use asarray() to convert potential matrix to ndarray
        self._translation = np.asarray(t).flatten()

    def to_matrix(self) -> npt.NDArray:
        """Return the matrix representation of this element.

        :return: 3x3 SE(2) matrix
        """
        T = np.identity(3)
        T[0:2, 0:2] = self.rotation.to_matrix()
        T[0:2, 2] = self.translation
        return T

    def to_tuple(self) -> Tuple[SO2, npt.NDArray]:
        """Return the tuple representation of this pose

        :return: (R (2x2 matrix), t (2D column vector)
        """
        return (self.rotation.to_matrix(), self.translation)

    def Log(self) -> npt.NDArray:
        """Computes the tangent space vector xi_vec at the current element X.

        :return: The tangent space vector xi_vec = [rho_vec, theta]^T.
        """
        theta = self.rotation.Log()

        if theta == 0:
            return np.hstack((self.translation, 0))

        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / theta

        V_inv = (1.0 / (a**2 + b**2)) * np.array([[a, b], [-b, a]])
        rho_vec = V_inv @ self.translation

        return np.hstack((rho_vec, theta))

    def jac_inverse_X_wrt_X(X) -> npt.NDArray:
        """Computes the Jacobian of the inverse operation X.inverse() with respect to the element X.

        :return: The Jacobian (3x3 matrix)
        """
        return -X.adjoint()

    def jac_action_Xx_wrt_X(X, x: npt.NDArray) -> npt.NDArray:
        """Computes the Jacobian of the action X.action(x) with respect to the element X.

        :param x: The 2D column vector x.
        :return: The Jacobian (3x2 matrix)
        """
        return np.block([[X.rotation.to_matrix(), 
                          X.rotation.to_matrix() @ SO2.hat(1) @ x[:, None]]])

    def jac_action_Xx_wrt_x(X) -> npt.NDArray:
        """Computes the Jacobian of the action X.action(x) with respect to the element X.

        :return: The Jacobian (2x2 matrix)
        """
        return X.rotation.to_matrix()

    def jac_Y_ominus_X_wrt_X(Y, X: SE2) -> npt.NDArray:
        """Compute the Jacobian of Y.ominus(X) with respect to the element X.

        :param X: The SE(2) element X.
        :return: The Jacobian (3x3 matrix)
        """
        return -SE2.jac_left_inverse(Y - X)

    def jac_Y_ominus_X_wrt_Y(Y, X: SE2) -> npt.NDArray:
        """Compute the Jacobian of Y.ominus(X) with respect to the element Y.

        :param X: The SE(2) element X.
        :return: The Jacobian (3x3 matrix)
        """
        return SE2.jac_right_inverse(Y - X)

    def inverse(self) -> SE2:
        """Compute the inverse of the current element X.

        :return: The inverse of the current element.
        """
        rot_inv = self.rotation.inverse()
        return SE2(rot_inv, -(rot_inv * self.translation))

    def action(self, x: npt.NDArray) -> npt.NDArray:
        """Perform the action of the SE(2) element on the 2D column vector x.

        :param x: 2D column vector to be transformed (or a matrix of 2D column vectors)
        :return: The resulting rotated and translated 2D column vectors
        """
        return self.rotation * x + self.translation

    def compose(X, Y: SE2) -> SE2:
        """Compose this element with another element on the right

        :param Y: The other SE2 element
        :return: This element composed with Y
        """
        return SE2(X.rotation @ Y.rotation, X.rotation * Y.translation + X.translation)

    def adjoint(self) -> npt.NDArray:
        """The adjoint at the element.
        :return: The adjoint, a 3x3 matrix.
        """
        R = self.rotation.to_matrix()
        t = self.translation

        return np.block([[R, -SO2.hat(1.0) @ t[:, None]],
                         [np.zeros((1, 2)), 1.0]])

    def oplus(X, xi_vec: npt.NDArray) -> SE2:
        """Computes the right perturbation of Exp(theta_vec) on the element X.

        :param theta_vec: The tangent space vector, a 3D column vector.
        :return: The perturbed SE2 element Y = X :math:`\\oplus` theta_vec.
        """

        if not (isinstance(xi_vec, np.ndarray) and xi_vec.shape == (3,)):
            raise TypeError('Argument must be a 3D column vector')

        return X @ SE2.Exp(xi_vec)

    def ominus(Y, X: SE2) -> npt.NDArray:
        """Computes the tangent space vector at X between X and this element Y.

        :param X: The other element
        :return: The difference xi_vec = Y :math:'\\ominus' X
        """
        if not isinstance(X, SE2):
            raise TypeError('Argument must be an SE2')
        return (X.inverse() @ Y).Log()

    def __add__(self, theta_vec: npt.NDArray) -> SE2:
        """Add operator performs the "oplus" operation on the element X.

        :param theta_vec: The tangent space vector, a 3D column vector.
        :return: The perturbed SE3 element Y = X :math:`\\oplus` theta_vec.
        """
        return self.oplus(theta_vec)

    def __sub__(self, X: SE2) -> npt.NDArray:
        """Subtract operator performs the "ominus" operation at X between X and this element Y.

        :param X: The other element
        :return: The difference theta_vec = Y :math:'\\ominus' X
        """
        return self.ominus(X)

    def __mul__(self, other: SE2) -> npt.NDArray:
        """Multiplication operator performs action on vectors.

        :param other: 2D column vector, or a matrix of 2D column vectors
        :return: Transformed 2D column vectors
        """
        if isinstance(other, np.ndarray) and other.shape[0] == 2:
            # Other is matrix of 2D column vectors, perform action on vectors.
            return self.action(other)
        else:
            raise TypeError('Argument must be a matrix of 3D column vectors')

    def __matmul__(self, other: SE2) -> SE2:
        """Matrix multiplication operator performs composition on elements of SE(2).

        :param other: Other SE2
        :return: Composed SE2
        """
        if isinstance(other, SE2):
            # Other is SE2, perform composition.
            return self.compose(other)
        else:
            raise TypeError('Argument must be an SE2')

    def __len__(self) -> int:
        """Length operator returns the dimension of the tangent vector space,
        which is equal to the number of degrees of freedom (DOF).

        :return: The DOF for SE(2) (3)
        """
        return 3

    def __repr__(self) -> str:
        """Formal string representation of the object.

        :return: The formal representation as a string
        """
        return "SE2(\n" + repr(self.to_matrix()) + "\n)"

    def __str__(self) -> str:
        """Informal string representation of the object
        prints the matrix representation.

        :return: The matrix representation as a string
        """
        return str(self.to_matrix())


    @staticmethod
    def hat(xi_vec: npt.NDArray) -> npt.NDArray:
        """Performs the hat operator on the tangent space vector xi_vec,
        which returns the corresponding Lie Algebra matrix xi_hat.

        :param xi_vec: 3D tangent space column vector xi_vec = [rho_vec, theta]^T.
        :return: The Lie Algebra (3x3 matrix).
        """
        return np.block([[SO2.hat(xi_vec[2].item()), xi_vec[:2, None]],
                         [np.zeros((1, 3))]])

    @staticmethod
    def vee(xi_hat: npt.NDArray) -> npt.NDArray:
        """Performs the vee operator on the Lie Algebra matrix xi_hat,
        which returns the corresponding tangent space vector.

        :param xi_hat: The Lie Algebra (3x3 matrix)
        :return: 3D tangent space column vector xi_vec = [rho_vec, theta]^T.
        """
        return np.hstack((xi_hat[:2, 2], SO2.vee(xi_hat[:2, :2])))

    @staticmethod
    def Exp(xi_vec: npt.NDArray) -> SE2:
        """Computes the Exp-map on the Lie algebra vector xi_vec,
        which transfers it to the corresponding Lie group element.

        :param xi_vec: 3D tangent space column vector xi_vec = [rho_vec, theta]^T.
        :return: Corresponding SE(2) element
        """
        rho_vec = xi_vec[:2]
        theta = xi_vec[2].item()

        if np.abs(theta) < 1e-10:
            return SE2(SO2(theta), rho_vec)

        V = (np.sin(theta) / theta) * np.identity(2) + ((1 - np.cos(theta)) / theta) * SO2.hat(1)
        return SE2(SO2(theta), V @ rho_vec)

    @staticmethod
    def jac_composition_XY_wrt_X(Y: SE2) -> npt.NDArray:
        """Computes the Jacobian of the composition X.compose(Y) with respect to the element X.

        :param Y: SE2 element Y
        :return: The Jacobian (3x3 matrix)
        """
        R_Y_inv = Y.rotation.inverse().to_matrix()
        return np.block([[R_Y_inv, (R_Y_inv @ SO2.hat(1.0) @ Y.translation[:, None])],
                         [np.array([0, 0, 1])]])

    @staticmethod
    def jac_composition_XY_wrt_Y() -> npt.NDArray:
        """Computes the Jacobian of the composition X.compose(Y) with respect to the element Y.

        :return: The Jacobian (3x3 identity matrix)
        """
        return np.identity(3)


    @staticmethod
    def jac_right(xi_vec: npt.NDArray) -> npt.NDArray:
        """Compute the right derivative of Exp(xi_vec) with respect to xi_vec.

        :param xi_vec: The tangent space 3D column vector xi_vec = [rho_vec, theta]^T.
        :return: The Jacobian (3x3 matrix)
        """
        rho_1, rho_2, theta = xi_vec.flatten()

        theta_inv = 1.0 / theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        J_r = np.array([[sin_theta * theta_inv, (1 - cos_theta) * theta_inv, (theta * rho_1 - rho_2 + rho_2 * cos_theta - rho_1 * sin_theta) * theta_inv**2],
                        [(cos_theta - 1) * theta_inv, sin_theta * theta_inv, (rho_1 + theta * rho_2 - rho_1 * cos_theta - rho_2 * sin_theta) * theta_inv**2],
                        [0, 0, 1]])

        return J_r

    @staticmethod
    def jac_left(xi_vec: npt.NDArray) -> npt.NDArray:
        """Compute the left derivative of Exp(xi_vec) with respect to xi_vec.

        :param xi_vec: The tangent space 3D column vector xi_vec = [rho_vec, theta]^T.
        :return: The Jacobian (3x3 matrix)
        """
        rho_1, rho_2, theta = xi_vec.flatten()

        theta_inv = 1.0 / theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        J_l = np.array([[sin_theta * theta_inv, (cos_theta - 1) * theta_inv, (theta * rho_1 + rho_2 - rho_2 * cos_theta - rho_1 * sin_theta) * theta_inv**2],
                        [(1 - cos_theta) * theta_inv, sin_theta * theta_inv, (-rho_1 + theta * rho_2 + rho_1 * cos_theta - rho_2 * sin_theta) * theta_inv**2],
                        [0, 0, 1]])

        return J_l

    @staticmethod
    def jac_right_inverse(xi_vec: npt.NDArray) -> npt.NDArray:
        """Compute the right derivative of Log(X) with respect to X for xi_vec = Log(X).

        :param xi_vec: The tangent space 3D column vector xi_vec = [rho_vec, theta]^T.
        :return: The Jacobian (3x3 matrix)
        """
        return np.linalg.inv(SE2.jac_right(xi_vec))

    @staticmethod
    def jac_left_inverse(xi_vec: npt.NDArray) -> npt.NDArray:
        """Compute the left derivative of Log(X) with respect to X for xi_vec = Log(X).

        :param xi_vec: The tangent space 3D column vector xi_vec = [rho_vec, theta]^T.
        :return: The Jacobian (3x3 matrix)
        """
        return np.linalg.inv(SE2.jac_left(xi_vec))

    @staticmethod
    def jac_X_oplus_tau_wrt_X(xi_vec: npt.NDArray) -> npt.NDArray:
        """Compute the Jacobian of X.oplus(tau) with respect to the element X

        :param xi_vec: The tangent space 3D column vector xi_vec = [rho_vec, theta]^T.
        :return: The Jacobian (3x3 matrix)
        """
        return SE2.Exp(xi_vec).inverse().adjoint()

    @staticmethod
    def jac_X_oplus_tau_wrt_tau(xi_vec: npt.NDArray) -> npt.NDArray:
        """Compute the Jacobian of X.oplus(tau) with respect to the tangent space vector tau

        :param xi_vec: The tangent space 3D column vector xi_vec = [rho_vec, theta]^T.
        :return: The Jacobian (3x3 matrix)
        """
        return SE2.jac_right(xi_vec)
