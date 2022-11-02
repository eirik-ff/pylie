import numpy as np
from pylie import SO2, SE2


def test_construct_identity_with_no_args():
    se2 = SE2()
    np.testing.assert_equal(se2.rotation.angle, 0.0)
    np.testing.assert_equal(se2.rotation.to_matrix(), np.identity(2))
    np.testing.assert_equal(se2.translation, np.zeros((2,)))


def test_construct_with_tuple():
    so2 = SO2(0.1)
    t = np.array([1, 2])
    se2 = SE2(so2, t)

    np.testing.assert_equal(se2.rotation.to_matrix(), so2.to_matrix())
    np.testing.assert_equal(se2.translation, t)


def test_construct_with_matrix():
    so2 = SO2(np.pi / 10)
    t = np.array([1, 2])
    T = np.block([[so2.to_matrix(), t[:, None]],
                  [0, 0,  1]])

    se2 = SE2.from_matrix(T)

    np.testing.assert_almost_equal(se2.rotation.to_matrix(), so2.to_matrix(), 14)
    np.testing.assert_equal(se2.translation, t)


def test_to_matrix():
    so2 = SO2(np.pi / 4)
    t = np.array([3, 2])
    se2 = SE2(so2, t)
    T = se2.to_matrix()

    np.testing.assert_equal(T[0:2, 0:2], so2.to_matrix())
    np.testing.assert_equal(T[0:2, 2], t)
    np.testing.assert_equal(T[2, :], np.array([0, 0, 1]))


def test_to_tuple():
    so2 = SO2(-np.pi / 3)
    t = np.array([-1, 1])
    se2 = SE2(so2, t)
    pose_tuple = se2.to_tuple()

    np.testing.assert_equal(pose_tuple[0], so2.to_matrix())
    np.testing.assert_equal(pose_tuple[1], t)


def test_composition_with_identity():
    X = SE2(SO2(np.pi / 10), np.array([3, 1]))

    comp_with_identity = X.compose(SE2())
    comp_from_identity = SE2().compose(X)

    np.testing.assert_almost_equal(comp_with_identity.to_matrix(), X.to_matrix(), 14)
    np.testing.assert_almost_equal(comp_from_identity.to_matrix(), X.to_matrix(), 14)


def test_composition():
    X = SE2(SO2(-np.pi / 2), np.array([1, 2]))
    Y = SE2(SO2(3 * np.pi / 2), np.array([0, 1]))

    Z = X.compose(Y)

    rot_expected = SO2(np.pi)
    t_expected = np.array([2, 2])

    np.testing.assert_almost_equal(Z.rotation.to_matrix(), rot_expected.to_matrix(), 14)
    np.testing.assert_almost_equal(Z.translation, t_expected, 14)


def test_composition_with_operator():
    X = SE2(SO2(np.pi), np.array([1, 2]))
    Y = SE2(SO2(-np.pi / 2), np.array([-1, 0]))

    Z = X @ Y

    rot_expected = SO2(np.pi / 2)
    t_expected = np.array([2, 2])

    np.testing.assert_almost_equal(Z.rotation.to_matrix(), rot_expected.to_matrix(), 14)
    np.testing.assert_almost_equal(Z.translation, t_expected, 14)


def test_inverse():
    X = SE2(SO2(np.pi / 4), np.array([1, -2]))

    X_inv = X.inverse()

    np.testing.assert_equal(X_inv.rotation.to_matrix(), X.rotation.inverse().to_matrix())
    np.testing.assert_equal(X_inv.translation, -(X.rotation.inverse() * X.translation))

    np.testing.assert_almost_equal((X @ X_inv).to_matrix(), np.identity(3))


def test_action_on_vectors():
    unit_x = np.array([1, 0])
    unit_y = np.array([0, 1])
    t = np.array([3, 1])

    X = SE2(SO2(np.pi / 2), t)

    np.testing.assert_almost_equal(X.action(unit_x), unit_y + t, 14)


def test_action_on_vectors_with_operator():
    X = SE2(SO2(3 * np.pi / 2), np.array([1, 2]))
    x = np.array([-1, 0])

    vec_expected = np.array([1, 3])

    np.testing.assert_almost_equal(X * x, vec_expected, 14)


def test_hat():
    rho_vec = np.array([1, 2])
    theta = np.pi / 3
    xi_vec = np.hstack((rho_vec, theta))

    xi_hat_expected = np.block([[SO2.hat(theta), rho_vec[:, None]],
                                [np.zeros((1, 3))]])

    np.testing.assert_equal(SE2.hat(xi_vec), xi_hat_expected)


def test_vee():
    rho_vec = np.array([4, 3])
    theta = 2 * np.pi / 3
    xi_vec = np.hstack((rho_vec, theta))
    xi_hat = SE2.hat(xi_vec)

    np.testing.assert_equal(SE2.vee(xi_hat), xi_vec)


def test_adjoint():
    np.testing.assert_equal(SE2().adjoint(), np.identity(3))

    X = SE2(SO2(3 * np.pi / 2), np.array([1, 2]))
    Adj = X.adjoint()

    np.testing.assert_almost_equal(Adj[:2, :2], X.rotation.to_matrix(), 14)
    np.testing.assert_almost_equal(Adj[:2, 2], -SO2.hat(1.0) @ X.translation, 14)
    np.testing.assert_almost_equal(Adj[2, :], np.array([0, 0, 1]), 14)


def test_log_with_no_rotation():
    X = SE2(SO2(), np.array([-1, 2]))

    xi_vec = X.Log()

    # Translation part:
    np.testing.assert_equal(xi_vec[:2], X.translation)

    # Rotation part:
    np.testing.assert_equal(xi_vec[2].item(), 0)


def test_exp_with_no_rotation():
    rho_vec = np.array([1, 2])
    theta = 0.0
    xi_vec = np.hstack((rho_vec, theta))

    X = SE2.Exp(xi_vec)

    np.testing.assert_equal(X.rotation.to_matrix(), np.identity(2))
    np.testing.assert_equal(X.translation, rho_vec)


def test_log_of_exp():
    rho_vec = np.array([2, 3])
    theta = 2 * np.pi / 3
    xi_vec = np.hstack((rho_vec, theta))

    X = SE2.Exp(xi_vec)

    np.testing.assert_almost_equal(X.Log(), xi_vec, 14)


def test_exp_of_log():
    X = SE2(SO2(np.pi / 10), np.array([2, 1]))
    xi_vec = X.Log()

    np.testing.assert_almost_equal(SE2.Exp(xi_vec).to_matrix(), X.to_matrix(), 14)


def test_ominus_with_oplus_diff():
    X = SE2(SO2(np.pi / 10), np.array([3, 2]))

    rho_vec = np.array([4, 3])
    theta = 2 * np.pi / 3
    xi_vec = np.hstack((rho_vec, theta))

    Y = X.oplus(xi_vec)
    xi_vec_diff = Y.ominus(X)

    np.testing.assert_almost_equal(xi_vec_diff, xi_vec, 14)


def test_oplus_with_ominus_diff():
    X = SE2(SO2(-np.pi / 5), np.array([2, 1]))
    Y = SE2(SO2(np.pi / 7), np.array([1, 0]))

    xi_vec_diff = Y.ominus(X)
    Y_from_X = X.oplus(xi_vec_diff)

    np.testing.assert_almost_equal(Y_from_X.to_matrix(), Y.to_matrix(), 14)


def test_ominus_with_oplus_diff_with_operators():
    X = SE2(SO2(np.pi / 7), np.array([1, 0]))

    rho_vec = np.array([1, 3])
    theta = 2 * np.pi / 3
    xi_vec = np.hstack((rho_vec, theta))

    Y = X + xi_vec
    xi_vec_diff = Y - X

    np.testing.assert_almost_equal(xi_vec_diff, xi_vec, 14)


def test_jacobian_inverse():
    X = SE2(SO2(np.pi / 7), np.array([2, 1]))

    J_inv = X.jac_inverse_X_wrt_X()

    # Jacobian should be -X.adjoint().
    np.testing.assert_equal(J_inv, -X.adjoint())

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = X.oplus(delta).inverse() - X.inverse().oplus(J_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 14)


def test_jacobian_action_Xx_wrt_X():
    X = SE2(SO2(np.pi / 8), np.array([1, 1]))
    x = np.array([1, 2])

    J_action_X = X.jac_action_Xx_wrt_X(x)

    # Jacobian should be [R,  R*SO3.hat(1)*x].
    np.testing.assert_array_equal(J_action_X, np.block([[X.rotation.to_matrix(), X.rotation.to_matrix() @ SO2.hat(1) @ x[:, None]]]))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = X.oplus(delta).action(x) - (X.action(x) + J_action_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((2,)), 5)


def test_jacobian_action_Xx_wrt_x():
    X = SE2(SO2(np.pi / 8), np.array([2, 1]))
    x = np.array([2, 3])

    J_action_x = X.jac_action_Xx_wrt_x()

    # Jacobian should be R.
    np.testing.assert_array_equal(J_action_x, X.rotation.to_matrix())

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((2,))
    taylor_diff = X.action(x + delta) - (X.action(x) + J_action_x @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((2,)), 14)


def test_jacobian_composition_XY_wrt_X():
    X = SE2(SO2(np.pi / 10), np.array([2, 1]))
    Y = SE2(SO2(np.pi / 7), np.array([1, 0]))

    J_comp_X = X.jac_composition_XY_wrt_X(Y)

    # Jacobian should be Y.inverse().adjoint()
    np.testing.assert_almost_equal(J_comp_X, Y.inverse().adjoint(), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = X.oplus(delta).compose(Y) - X.compose(Y).oplus(J_comp_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 14)


def test_jacobian_composition_XY_wrt_Y():
    X = SE2(SO2(np.pi / 10), np.array([3, 2]))
    Y = SE2(SO2(np.pi / 7), np.array([1, 0]))

    J_comp_Y = X.jac_composition_XY_wrt_Y()

    # Jacobian should be identity
    np.testing.assert_array_equal(J_comp_Y, np.identity(3))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = X.compose(Y.oplus(delta)) - X.compose(Y).oplus(J_comp_Y @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 14)


def test_jacobian_right():
    rho_vec = np.array([1, 2])
    theta = 3 * np.pi / 4
    xi_vec = np.hstack((rho_vec, theta))

    J_r = SE2.jac_right(xi_vec)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = SE2.Exp(xi_vec + delta) - (SE2.Exp(xi_vec) + J_r @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 5)


def test_jacobian_left():
    rho_vec = np.array([2, 1])
    theta = np.pi / 4
    xi_vec = np.hstack((rho_vec, theta))

    J_l = SE2.jac_left(xi_vec)

    # Should have J_l(xi_vec) == J_r(-xi_vec).
    np.testing.assert_almost_equal(J_l, SE2.jac_right(-xi_vec), 14)

    # Test the Jacobian numerically (using Exps and Logs, since left oplus and ominus have not been defined).
    delta = 1e-3 * np.ones((3,))
    taylor_diff = SE2.Log(SE2.Exp(xi_vec + delta) @ (SE2.Exp(J_l @ delta) @ SE2.Exp(xi_vec)).inverse())
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 5)


def test_jacobian_right_inverse():
    X = SE2(SO2(np.pi / 8), np.array([1, 1]))
    xi_vec = X.Log()

    J_r_inv = SE2.jac_right_inverse(xi_vec)

    # Should have J_l * J_r_inv = Exp(xi_vec).adjoint().
    J_l = SE2.jac_left(xi_vec)
    np.testing.assert_almost_equal(J_l @ J_r_inv, SE2.Exp(xi_vec).adjoint(), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = X.oplus(delta).Log() - (X.Log() + J_r_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 5)


def test_jacobian_left_inverse():
    X = SE2(SO2(np.pi / 8), np.array([2, 1]))
    xi_vec = X.Log()

    J_l_inv = SE2.jac_left_inverse(xi_vec)

    # Test the Jacobian numerically (using Exps and Logs, since left oplus and ominus have not been defined).
    delta = 1e-3 * np.ones((3,))
    taylor_diff = (SE2.Exp(delta) @ X).Log() - (X.Log() + J_l_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 5)


def test_jacobian_X_oplus_tau_wrt_X():
    X = SE2(SO2(np.pi / 10), np.array([3, 2]))
    rho_vec = np.array([1, 2])
    theta = np.pi / 4
    xi_vec = np.hstack((rho_vec, theta))

    J_oplus_X = X.jac_X_oplus_tau_wrt_X(xi_vec)

    # Should be Exp(tau).adjoint().inverse()
    np.testing.assert_almost_equal(J_oplus_X, np.linalg.inv(SE2.Exp(xi_vec).adjoint()), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = X.oplus(delta).oplus(xi_vec) - X.oplus(xi_vec).oplus(J_oplus_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 14)


def test_jacobian_X_oplus_tau_wrt_tau():
    X = SE2(SO2(np.pi / 8), np.array([1, 2]))
    rho_vec = np.array([2, 1])
    theta = np.pi / 4
    xi_vec = np.hstack((rho_vec, theta))

    J_oplus_tau = X.jac_X_oplus_tau_wrt_tau(xi_vec)

    # Should be J_r.
    np.testing.assert_equal(J_oplus_tau, X.jac_right(xi_vec))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = X.oplus(xi_vec + delta) - X.oplus(xi_vec).oplus(J_oplus_tau @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 6)


def test_jacobian_Y_ominus_X_wrt_X():
    X = SE2(SO2(np.pi / 8), np.array([1, 1]))
    Y = SE2(SO2(np.pi / 7), np.array([1, 0]))

    J_ominus_X = Y.jac_Y_ominus_X_wrt_X(X)

    # Should be -J_l_inv.
    np.testing.assert_equal(J_ominus_X, -SE2.jac_left_inverse(Y - X))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = Y.ominus(X.oplus(delta)) - (Y.ominus(X) + (J_ominus_X @ delta))
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 6)


def test_jacobian_Y_ominus_X_wrt_Y():
    X = SE2(SO2(np.pi / 8), np.array([1, 1]))
    Y = SE2(SO2(np.pi / 7), np.array([2, 0]))

    J_ominus_Y = Y.jac_Y_ominus_X_wrt_Y(X)

    # Should be J_r_inv.
    np.testing.assert_equal(J_ominus_Y, SE2.jac_right_inverse(Y - X))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3,))
    taylor_diff = Y.oplus(delta).ominus(X) - (Y.ominus(X) + (J_ominus_Y @ delta))
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3,)), 6)


def test_has_len_that_returns_correct_dimension():
    X = SE2()

    # Should have 6 DOF.
    np.testing.assert_equal(len(X), 3)


def test_string_representation_is_matrix():
    X = SE2()

    # Should be the same as the string representation of the matrix.
    np.testing.assert_equal(str(X), str(X.to_matrix()))
