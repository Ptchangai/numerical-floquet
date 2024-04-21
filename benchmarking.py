#This script will be used to compare algorithms: Speed, accuracy, stability
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

def compute_accuracy(method, function, param):
    """
    Measure accuracy of a numerical method for a given function
    """
    ...
    return

def compute_stability(method, function, param):
    """
    Measure stability of a numerical method for a given function
    """
    ...
    return