import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure
from differential_equations import lotka_volterra, n_bodies
from numerical_integration import solve_ode, solve_ode_adapt, RK4step, RK34step
from benchmarking import compute_execution_time
import numpy as np


#Main operations will be done here:
def test_LV():
    initial_conditions = [1, 4]  # Initial values for [x, y]
    step_size = 0.001
    num_iterations = 2000
    param = (3, 9, 15, 15)
    result1 = solve_ode(initial_conditions, step_size, num_iterations, ode_func=lotka_volterra, param=param, stepper_func=RK4step)
    result2 = solve_ode_adapt(initial_conditions, step_size, num_iterations, ode_func=lotka_volterra, param=param, stepper_func=RK34step)
    lotka1 = [i[0] for i in result1]
    volterra1 = [i[1] for i in result1]
    plt.plot(lotka1,volterra1)
    lotka2 = [i[0] for i in result2]
    volterra2 = [i[1] for i in result2]
    plt.plot(lotka2, volterra2)
    print("Lotka Volterra test passed")

#TODO: complete last lines of test_planet to fetch and plot results.
#TODO: Find initial values and parameter values that actually work.
#TODO: Add benchmark comparison to tests
def test_planets():
    """
    Run basic test for N_body ODE with solvers.
    """
    initial_conditions = [0, 0, 0, 10, 0, 0,    # Velocity and position for body 1
                          0, 0, 0, -10, 0, 0]   # Velocity and position for body 2
    params = {
        'gravitational_constant': 6.67430e-11,  # Gravitational constant in m^3 kg^-1 s^-2
        'mass': [5.972e24, 5.972e24],           # Masses of two planets in kg
        'num_bodies': 2,
        'dimensions': 3}

    step_size = 0.001
    num_iterations = 2000
    result = solve_ode(initial_conditions, step_size, num_iterations, ode_func=n_bodies, param=params, stepper_func=RK4step)
    body1_x = [i[3] for i in result]
    body1_y = [i[4] for i in result]
    plt.plot(body1_x, body1_y, label='Body 1 Trajectory')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('N-Body Simulation Trajectory')
    plt.legend()
    plt.show()
    print("N_body test passed")
    return

def run_test():
    try:
        test_LV()
        test_planets()
    except Exception as e:
        print(e)
    return


##TODO: it would be interesting if this section actually showed examples of bifurcations, etc..
if __name__ == "__main__":
    run_test()