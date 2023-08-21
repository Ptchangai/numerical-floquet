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



#Planetary movement?
#y: [vx0, vy0, Px0, Py0]
def planets(t,y):
  G = 6.67*10**(-11)
  G = 6.67*10**(-2)
  Ms = 1.98854*10**30
  Ms = 1.98854*10**3
  #Ps=[1.81899*10**8,9.83630*10**8,-1.5877*10**7]
  Ps = [0,0,0]
  d = np.sqrt((y[0]-Ps[0])**2+(y[1]-Ps[1])**2+(y[2]-Ps[2])**2)
  S = [0,0,0]
  S[0] = G*Ms*(Ps[0]-y[0])/d**3
  S[1] = G*Ms*(Ps[1]-y[1])/d**3
  S[2] = G*Ms*(Ps[2]-y[2])/d**3
  R = [y[3],y[4],y[5],S[0],S[1],S[2]]
  return R 