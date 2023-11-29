#This script will be used to compare algorithms: Speed, accuracy, stability
import time

def compute_execution_time(function):
    """Computes the time it takes to run function()"""
    start_time = time.time()
    function()  
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    return

#Measure accuracy
def compute_accuracy(function):
    ...
    return

#Measure stability
def compute_stability(function):
    ...
    return