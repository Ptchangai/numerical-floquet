import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure
from differential_equations import lotka_volterra, n_bodies
from numerical_integration import solve_ode, RK4step, RK34step
from benchmarking import compute_execution_time
import numpy as np


#Main operations will be done here:
def test_LV():
    initial_value = [1, 4]  # Initial values for [x, y]
    step_size = 0.001
    num_iterations = 2000
    param = (3, 9, 15, 15)
    result1 = solve_ode(initial_value, step_size, num_iterations, ode_func=lotka_volterra, param=param, stepper_func=RK4step)
    result2 = solve_ode(initial_value, step_size, num_iterations, ode_func=lotka_volterra, param=param, stepper_func=RK34step)
    lotka1 = [i[0] for i in result1]
    volterra1 = [i[1] for i in result1]
    plt.plot(lotka1,volterra1)
    lotka2 = [i[0] for i in result2]
    volterra2 = [i[1] for i in result2]
    plt.plot(lotka2, volterra2)

#TODO: complete last lines of test_planet to fetch and plot results.
#TODO: Find initial values and parameter values that actually work.
#TODO: Add benchmark comparison to tests
def test_planets():
    """
    Run basic test for N_boody ODE with solvers."""
    initial_value = [1, 4, 3,
                     2, 2 , 2]
    param = (...)

    step_size = 0.001
    num_iterations = 2000
    result1 = solve_ode(initial_value, step_size, num_iterations, ode_func=n_bodies, param=param, stepper_func=RK4step)
    body1 = ...
    plt.plot(body1[...], body1[...])

    return

def run_test():
    try:
        test_LV()
        print("Lotka Volterra test passed")
    except Exception as e:
        print(e)
    return

##TODO: it would be interesting if this section actually showed examples of bifurcations, etc..
if __name__ == "main":
    run_test()