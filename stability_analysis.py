import numpy as np
def geometric_poincare(ode_func, section, h, y0, params):
    """
    For a 2D ODE, approximate the first return time using some section of the orbit.
    """
    tol = 1e-5
    t = np.arange(0, 20+h, h)
    approx = np.zeros((len(t), len(y0)))
    approx[0] = y0
    number_intersection = 0
    intersection_list = []

    for i in range(len(t)-1):
        approx[i+1] = RK4step(ode_func, approx[i], t[i], h, params)
        x = segment_intersection(section[0], section[1], approx[i,:], approx[i+1,:])
        if x:
            print("Intersection at:", x)
            intersection_list.append(x)
            number_intersection += 1
            if number_intersection == 5:
                break
    
    plt.plot(section[:,0], section[:,1])
    plt.plot(approx[:,0], approx[:,1])
    for intersection in intersection_list:
        plt.plot(intersection[0], intersection[1], 'x')
    intersection_list = np.array(intersection_list)
    if np.linalg.norm([np.linalg.norm(intersection_list[0] - intersection_list[-1])]) < tol:
        print('Periodic solution')
    else:
        if intersection_list[0,1] < intersection_list[2,1]:
            print('Non-periodic source solution')
        elif intersection_list[0,1] > intersection_list[2,1]:
            print('Non-periodic sink solution')
    return

def jacobian_matrix(func, x, params):
    """
    Compute the Jacobian of function func at point x.
    """
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


def power_iteration(A, n=50):
    """
    Compute the largest eigenvalue of the matrix A.
    """
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

#TODO
def monodromy_matrix(diff_eq_func, jacobian_func, t):
    ...
    return

#TODO
def monodromy_matrix_shooting(diff_eq_func, jacobian_func, t):
    """
    Alternative method to compute the Monodromy matrix using shooting methods.
    """
    ...
    return