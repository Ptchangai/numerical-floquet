#Define numemrical integration methods to solve differential equaions
#Use assimulo.
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from assimulo.solvers import RungeKutta34
import numpy as np 

def RK4step(f,uold, told, h, param=None):
  """Runge kutta 4, single step"""
  uold = np.array(uold)
  unew = []
  for i in range(0,len(uold)):
    yp1 = f(told, uold, param)[i]
    yp2 = f(told + h/2, uold + h*yp1/2, param)[i]
    yp3 = f(told + h/2, uold + h*yp2/2, param)[i]
    yp4 = f(told + h, uold + h*yp3, param)[i]
    unew.append(uold[i] + h*(yp1 + 2*yp2 + 2*yp3 + yp4)/6)
  return unew

def RK34step(f,uold, told, h, param=None):
  """Runge kutta 34, single step"""
  uold = np.array(uold)
  unew = []
  for i in range(0,len(uold)):
    yp1 = f(told, uold, param)[i]
    yp2 = f(told + h/2, uold + h*yp1/2, param)[i]
    yp3 = f(told + h/2, uold + h*yp2/2, param)[i]
    zp3 = f(told + h/2, uold - h*yp1 + 2*h*yp2)
    yp4 = f(told + h, uold + h*yp3, param)[i]
    error_estimate = h*(2*yp2+zp3-2*yp3-yp4)/6
    error = np.max(error_estimate)
    unew.append(uold[i] + h*(yp1 + 2*yp2 + 2*yp3 + yp4)/6)
  return unew, error

def newstep(tol, error, erro_old, h_old, k):
    m_error = np.max(error)
    m_error_old = np.max(m_error_old)
    r = np.linalg.norm(m_error)
    r_old = np.linalg.norm(m_error_old)
    hnew = (tol/r)^(2/(3*k))*(tol/r_old)^(-1/(3*k))*h_old
    return hnew


def solve_ode(initial_value, step_size, num_iterations, ode_func, param=None, stepper_func=RK4step):
    t_values = np.arange(0, step_size*num_iterations, step_size)
    u_values = [initial_value]
    u_current = initial_value
    for i in range(num_iterations):
        u_new = stepper_func(ode_func, u_current, t_values[i], step_size, param)
        u_values.append(u_new)
        u_current = u_new
    return u_values


def lagrange_polynomials(x,xm,i):
  """Computes Lagrange polynomial for interpolation"""
  n = len(xm)
  product = 1
  for j in range(n):
    if i != j:
       product *= (x-xm[j])/(xm[i]-xm[j])
  return product

#Define interplation algorithm
def interpolation(x,xm,ym):
  Poly = 0
  for i in range(len(xm)):
    Poly += ym[i]*lagrange_polynomials(x,xm,i)
  return Poly

def collocation_methods():
    ...
    return

def collocation_solve(F, y0, t_span, num_collocation_points):
    t_start, t_end = t_span
    t_collocation = np.linspace(t_start, t_end, num_collocation_points)
    num_dimensions = len(y0)
    y_collocation = np.zeros((num_collocation_points, num_dimensions))
    
    # Initial guess for the y values at collocation points
    y_guess = np.outer(np.ones(num_collocation_points), y0)
    
    def residual(y_colloc_flat):
        y_colloc = y_colloc_flat.reshape(num_collocation_points, num_dimensions)
        residuals = []
        for i in range(num_collocation_points):
            residual_i = np.zeros(num_dimensions)
            for j in range(num_dimensions):
                y_interp = interpolation(t_collocation[i], t_collocation, y_colloc[:, j])
                residual_i[j] = y_colloc[i, j] - y_interp
            residuals.extend(residual_i)
        return residuals
    
    y_colloc_flat = fsolve(residual, y_guess.flatten())
    y_collocation = y_colloc_flat.reshape(num_collocation_points, num_dimensions)
    
    return t_collocation, y_collocation


def shooting_methods():
    v0, = fsolve(objective_shooting, v0)
    return v0

def objective_shooting(v0, func):
    sol = solve_ivp(func, [t0, t1], \
            [y0, v0], t_eval = t_eval)
    y = sol.y[0]
    return y[-1] - 50