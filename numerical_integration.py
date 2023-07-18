#Define numemrical integration methods to solve differential equaions
#Use assimulo.
from scipy.optimize import fsolve, solve_ivp
from assimulo.solvers import RungeKutta34
import numpy as np 

def RK4step(f,uold, told, h):
  uold=np.array(uold)
  unew=[]
  for i in range(0,len(uold)):
    yp1=f(told, uold)[i]
    yp2=f(told + h/2, uold + h*yp1/2)[i]
    yp3=f(told + h/2, uold + h*yp2/2)[i]
    yp4=f(told + h, uold + h*yp3)[i]
    unew.append (uold[i] + h*(yp1 + 2*yp2 + 2*yp3 + yp4)/6)
  return unew

def runge_kutta():
    ...
    return


def lagrange_polynomials(x,xm,i):
  n=len(xm)
  Li=1
  for j in range(n) :
    if i!=j:
       Li*=(x-xm[j])/(xm[i]-xm[j])
  return Li

#Define interplation algorithm
def interpolation(x,xm,ym):
  Poly=0
  for i in range(len(xm)):
    Poly+=ym[i]*lagrange_polynomials(x,xm,i)
  return Poly

def collocation_methods():
    ...
    return

def shooting_methods():
    v0, = fsolve(objective_shooting, v0)
    return v0

def objective_shooting(v0, func):
    sol = solve_ivp(func, [t0, t1], \
            [y0, v0], t_eval = t_eval)
    y = sol.y[0]
    return y[-1] - 50