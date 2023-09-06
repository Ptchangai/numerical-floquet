#Define the differential equations on which we will test the Floquet algorithms 
import numpy as np
from assimulo.problem import Explicit_Problem

#mod = Explicit_Problem(rhs, y0, t0)


def exponential(t,y,param=5):
    r = param
    return (r*y)

def sinusoidal(t,y):
    R = [y[1], -y[0]]
    return R

#Harvesting constant p
def logistic(t,y,param=(1,2,0)):
  (a,k,p) = param
  return a*y*(1-y/k)-p


def lotka_volterra(t,y,param=(3,9,15,15)):
  (a,b,c,d) = param
  R=[a*y[0]-b*y[0]*y[1],c*y[0]*y[1]-d*y[1]]
  return R 

def van_der_pol(t,y,param=0.1):
  e=param
  R=[y[1],e*(1-y[0]**2)*y[1]-y[0]]
  return R

#Define the functions
def sincos(t,y):
  R=[-np.sin(y[1]),np.cos(y[0])]
  return R



#Planetary movement
#y: [vx0, vy0, Px0, Py0]
def planets(t,y, param):

  G = param['gravitational_constant']
  mass = param['mass']
  y1_pos = y[:3]  # Position of the first body
  y2_pos = y[6:9] # Position of the second body

  d = np.sqrt(np.sum((y1_pos - y2_pos)**2))   # distance between the first body and the second body
  gravitational_force = G * (y2_pos - y1_pos) / d**3 # gravitational force acting on the first body due to the second body
  S1 = mass[1] * gravitational_force
  S2 = -mass[0] * gravitational_force #opposite direction of forces. 

  #Equation of motions give:
  R1 = [y[3], y[4], y[5], S1[0], S1[1], S1[2]] #for the first body
  R2 = [y[9], y[10], y[11], S2[0], S2[1], S2[2]] #for the second body
  return R1 + R2