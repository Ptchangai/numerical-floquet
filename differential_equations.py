#Define the differential equations on which we will test the Floquet algorithms 
import numpy as np
from assimulo.problem import Explicit_Problem

#mod = Explicit_Problem(rhs, y0, t0)

def exponential(t, y, param=5):
    """
    Describes y = r*e^x.
    """
    r = param
    return (r*y)

def sinusoidal(t,y):
    R = [y[1], -y[0]]
    return R

#Harvesting constant p
def logistic(t, y, param=(1,2,0)):
  """
  Logistic growth with harvesting constant p.
  """
  (a,k,p) = param
  return a*y*(1-y/k)-p


def lotka_volterra(t,y,param=(3,9,15,15)):
  (a,b,c,d) = param
  R = [a*y[0]-b*y[0]*y[1], c*y[0]*y[1]-d*y[1]]
  return R 

def van_der_pol(t, y, param=0.1):
  """
  Describes a Van der Pol oscillator.
  """
  e = param
  R = [y[1], e*(1-y[0]**2)*y[1]-y[0]]
  return R

#Define the functions
def sincos(t,y):
  R = [-np.sin(y[1]), np.cos(y[0])]
  return R


def two_planets(t, y, param):
  """
  Planetary movement with two bodies a and b.
  y: [speed_x_a, speed_y_a, speed_z_a, position_x_a, position_y_a, position_z_a, 
      speed_x_b, speed_y_b, speed_z_b, position_x_b, position_y_b, position_z_b ] 
  params: dictionary with values for gravitational constant and mass.   
  """
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


def n_bodies(t, y, param):
      speed_x_a2, speed_y_a2, speed_z_a2, position_x_a2, position_y_a2, position_z_a2,
       ... ] 
    params: dictionary with values for gravitational constant and mass.   
    """
    G = param['gravitational_constant']
    mass = param['mass']
    num_bodies = param['num_bodies']
    dimensions = param['dimensions']
    positions = y[:num_bodies*dimensions]
    velocities = y[num_bodies*dimensions:]

    accelerations = np.zeros_like(positions)
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                rel_pos = positions[i*dimensions:(i+1)*dimensions] - positions[j*dimensions:(j+1)*dimensions]
                distance = np.linalg.norm(rel_pos)
                force = -G * mass[i] * mass[j] * rel_pos / (distance ** 3)
                accelerations[i*dimensions:(i+1)*dimensions] += force

    dydt = np.concatenate((velocities, accelerations))
    return dydt
