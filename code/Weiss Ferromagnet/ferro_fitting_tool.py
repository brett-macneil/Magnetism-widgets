#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 2020
Last updated Wed Jul 29 2020

@author: Brett MacNeil
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants as cst
from scipy.optimize import fsolve


# Load M vs T data to fit
#filename = ''
#T, M = np.genfromtxt(filename, unpack=True)
#M /=1e3 # A/m to kA/m


# Set up figure
fig, ax = plt.subplots(1, 1)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Magnetization (kA/m)')


# Physical constants
mu0 = cst['vacuum mag. permeability'][0]  # N/A^2
muB = cst['Bohr magneton'][0]             # J/T
g = -cst['electron g factor'][0]          # Unitless
kB = cst['Boltzmann constant'][0]         # J/K

# Sublattice parameters
N = 1.341e28                              # Moment volume density 1/m^3 


# Function definitions
def coth(x):
    return 1/np.tanh(x)


def brillouin(y, J, eps=1e-3):
    # Cast y to numpy array if int or float
    if type(y) != np.ndarray:
        y = np.array(y)
 
    B = np.empty(y.shape)
    mask = np.abs(y) >= eps     
    B[mask] = ((2*J+1)/(2*J)*coth((2*J+1)*y[mask]/(2*J)) 
              - coth(y[mask]/(2*J))/(2*J))
    
    # First order approximation for small |y|<eps
    # Approximation avoids divergence at origin
    B[~mask] = ((2*J+1)**2/J**2/12-1/J**2/12)*y[~mask]
    
    return B


def mag_eq(M, lam, mu, T, Jeff, N):
    y = -mu0*mu*lam*M
    y /= kB*T
    return N*mu*brillouin(y, Jeff) - M


def get_mag(T, lam, mu, kilo=True):
    # Effective angular momentum quantum number
    # mu = g*muB*J
    Jeff = mu/(g*muB)
    numpoints = len(T)
    Mag = np.empty(numpoints)
    guess = N*mu
    
    for i in range(numpoints):
        Mag[i] = fsolve(mag_eq, x0=guess, args=(lam, mu, T[i], Jeff, N))
        guess = Mag[i]
    
    if kilo:
        Mag /= 1e3
        
    return Mag


def curie_temp(lam, mu):
    # 1st order expansion of Brillouin 
    # function to estimate Curie temperature.
    Jeff = mu/(g*muB)
    Tc = lam*mu0*N*Jeff*(Jeff+1)*(g*muB)**2
    Tc /= 3*kB
    return Tc