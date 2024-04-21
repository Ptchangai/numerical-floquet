# This script will be used to compare algorithms: speed, accuracy, stability, etc.

import time

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

#TODO: complete function.
def compute_accuracy(method, function, param):
    """
    Measure accuracy of a numerical method for a given function
    """
    ...
    return

#TODO: complete function.
"""
Measure stability of a numerical method for a given function"""
def compute_stability(method, function, param):
    ...
    return