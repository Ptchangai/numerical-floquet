# Numerical Floquet: algorithms for stability analysis of limit cycles with Poincaré theory.


====================


## Contents

* RungeKutta
* Shooting methods
* Jacobian
* Monodromy
* Visualisation


## Setup

Before running the scripts, you can install the necessary Python packages by calling the following command in a console:

```
pip install -r requirements.txt

```


If you still encounter problems installing Assimulo from pip, you can try instead creating a conda environment.

```
conda create -n Floquet
conda activate Floquet
conda install -c conda-forge assimulo
```

Finally, you can then run the command:

```

conda jupyter notebook presentation.ipynb

```

This should open the jupyter notebook file in a browser. You can run through the steps contained in this page. 

## .py file description

* The ODEs are defined in the file 'differential_equations.py'.
* The numerical methods for solving such ODEs are defined in 'numerical_integration.py'.
* The deep learning methods for replicating the numerical methods are defined in 'neural_floquet.py'.
* The numerical methods for stability analysis are dfined in 'stability_analysis.py'.
* Speed and accuracy comparison functions can be found in file 'benchmarking.py'.

