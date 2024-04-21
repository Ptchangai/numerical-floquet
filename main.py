import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure
from differential_equations import lotka_volterra
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
    print('ok')
    result2 = solve_ode(initial_value, step_size, num_iterations, ode_func=lotka_volterra, param=param, stepper_func=RK34step)
    lotka1 = [i[0] for i in result1]
    volterra1 = [i[1] for i in result1]
    plt.plot(lotka1,volterra1)
    lotka2 = [i[0] for i in result2]
    volterra2 = [i[1] for i in result2]

#TODO: complete last lines of test_planet to fetch and plot results.
#TODO: Find initial values and parameter values that actually work.
#TODO: Add benchmark comparison to tests
def test_planets():
    """
    Run basic test for N_boody ODE with solvers.
    """
    return
def main():
    test_LV()
    print("Lotka Volterra test passed")
    return

##TODO: it would be interesting if this section actually showed examples of bifurcations, etc..
if __name__ == "main":
