#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 2020
Last updated Wed Jun 24 2020

@author: Brett MacNeil
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, Button
from scipy.optimize import fsolve
from scipy.constants import physical_constants as cst


### Set up figure
###______________________________________________________________

# Generate figure
fig, ax = plt.subplots(1, 1)
plt.subplots_adjust(bottom=0.4)
plt.style.use('dark_background')
#plt.style.use('lab')

# Label axes
ax.set_xlabel('Temperature (K)', fontsize=16)
ax.set_ylabel(r'Magnetization (kA m$^{-1}$)', fontsize=16)
ax.grid(color='grey', linestyle='dotted', linewidth=1)


### Sliders and buttons
###______________________________________________________________

# Interaction constant
lam_loc = plt.axes([0.125, 0.2, 0.775, 0.05])
lam_min = 1
lam_max = 1000
lam_init = 500
lam_sl = Slider(lam_loc, label=r'$\lambda$', valmin=lam_min, valmax=lam_max, \
                valinit=lam_init)
lam_sl.label.set_size(16)

# Angular momentum
J_loc = plt.axes([0.125, 0.1, 0.775, 0.05])
J_min = 1/2
J_max = 31/2
J_init = 1/2
J_sl = Slider(J_loc, label=r'$J$', valmin=J_min, valmax=J_max, \
              valinit=J_init, valstep=1)
J_sl.label.set_size(16)


### Constants
###______________________________________________________________

numpoints = 1000                              # Number of points in solver

# Physical constants
mu0 = cst['vacuum mag. permeability'][0]      # Permeability of free space
me = cst['electron mass'][0]                  # Electron mass [kg]
h = cst['Planck constant'][0]                 # Planck constant [J-s]
hbar = cst['reduced Planck constant'][0]      # Reduced Planck constant
e = cst['atomic unit of charge'][0]           # Elementary charge [C]
muB = cst['Bohr magneton'][0]                 # Bohr Magneton
g = -cst['electron g factor'][0]              # G-factor
kB = cst['Boltzmann constant'][0]             # Boltzmann constant [J/K]

N = 1e28                                      # Moments per unit volume

Tc = lam_init * mu0 * N * g**2 * J_init * (J_init+1) * muB**2 / (3*kB)
                                              # Curie temperature


### Function definitions
###______________________________________________________________

def coth(x):
    return 1/np.tanh(x)


def brillouin(y, J):
    eps = 1e-3 # should be small
    y = np.array(y); B = np.empty(y.shape)
    m = np.abs(y)>=eps # mask for selecting elements 
    
    B[m] = (2*J+1)/(2*J)*coth((2*J+1)*y[m]/(2*J)) - coth(y[m]/(2*J))/(2*J)
    
    # First order approximation for small |y|<eps
    # Approximation avoids divergence at origin
    B[~m] = ((2*J+1)**2/J**2/12-1/J**2/12)*y[~m]
    
    return B


def equation(M, lam, T, J):
    return N*g*muB*J*brillouin(g*mu0*muB*J*lam*M/(kB*T), J ) - M


def get_mag(Tc, numpoints, lam, J, kilo=True):
    Tvec = np.linspace(0, np.ceil(Tc*1.1), numpoints)
    M = np.empty(Tvec.shape)
    guess = N*g*muB*J # Initial guess
    
    for i in range(numpoints):
        M[i] = fsolve(equation, x0=guess, args=(lam, Tvec[i], J))
        guess = M[i] # Update guess

    if kilo == True:
        M = M / 1e3

    return (Tvec, M)
