#Goal is to compare algorithms: Speed, accuracy, stability
import time

def compute_execution_time(function):
    start_time = time.time()
    function()  
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")