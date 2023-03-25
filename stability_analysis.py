import numpy as np

def power_iteration(A):
    e=lambda x: x.T@A@x
    la,_=np.linalg.eig(A)
    lmax = max(abs(la))
    x=np.ones((4,), dtype=float)
    for i in range(50):
        z=A@x
        x=z/np.linalg.norm(z)
        l= e(x)
    error = abs(l-lmax)
    return l, error