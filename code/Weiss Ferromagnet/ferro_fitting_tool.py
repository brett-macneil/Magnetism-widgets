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