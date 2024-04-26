# This script will be used to compare algorithms: speed, accuracy, stability, etc.

import time
import numpy as np

from numerical_integration import *

def compute_execution_time(method, function, param):
    """
    Compute the time it takes to run function()
    """
    start_time = time.time()
    function()  
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    return

def compute_accuracy(method, ODE, param, true_solution=None):
    """
    Measure accuracy of a numerical method for a given function.
    - method: chosen numerical solver, e.g euler_step
    - ODE: differential equation to solve, e.g exponential.
    - param: needed parameters for the given ODE.
    - true_solution: function that solves the ODE, if any.
    """
    initial_value = ...
    step_size = ...
    num_iterations = ...
    t_values = np.arange(0, step_size*num_iterations + step_size, step_size)
    numerical_solution =  solve_ode(initial_value, step_size, num_iterations, ODE, param, stepper_func=method)
    true_solution = true_solution(t_values, param)
    difference = np.abs(numerical_solution - true_solution)
    accuracy = np.mean(difference)
    return accuracy

#TODO: complete function.
def compute_stability(method, function, param):
    """
    Measure stability of a numerical method for a given function
    """
    ...
    return