import numpy as np

def geometric_poincare():
    ...
    return

def jacobian_matrix(func, x, params):
    """Compute the Jacobian of function func at point x"""
    epsilon = 1e-6
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        J[:, i] = (func(x_plus, params) - func(x_minus, params)) / (2 * epsilon)
    return J


def power_iteration(A,n=50):
    """Compute the largest eigenvalue of the matrix A"""
    e = lambda x: x.T@A@x
    la, _ = np.linalg.eig(A)
    lmax = max(abs(la))
    x = np.ones((4,), dtype=float)
    for i in range(n):
        z = A@x
        x = z/np.linalg.norm(z)
        l = e(x)
    error = abs(l-lmax)
    return l, error

def monodromy_matrix(diff_eq_func, jacobian_func, t):
    ...
    return

def monodromy_matrix_shooting(diff_eq_func, jacobian_func, t):
    """Alternative method to compute the Monodromy matrix using shooting methods"""
    ...
    return