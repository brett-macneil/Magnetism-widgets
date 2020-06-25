#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 2020
Last updated Thu Jun 25 2020

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
fig, [ax1, ax2] = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.4)
plt.style.use('dark_background')
#plt.style.use('lab')

# Label axes
ax1.set_xlabel(r'$M_{b}$ (kA m$^{-1}$)', fontsize=16)
ax1.set_ylabel(r'$M_{a}$ (kA m$^{-1}$)', fontsize=16)
ax1.grid(color='grey', linestyle='dotted', linewidth=1)

ax2.set_xlabel('Temperature (K)', fontsize=16)
ax2.set_ylabel(r'Magnetization (kA m$^{-1}$)', fontsize=16)
ax2.grid(color='grey', linestyle='dotted', linewidth=1)


### Constants
###______________________________________________________________

numpoints = 500           # Number of points used in equation solver

# Physical constants
mu0 = cst['vacuum mag. permeability'][0]      # Permeability of free space
me = cst['electron mass'][0]                  # Electron mass [kg]
h = cst['Planck constant'][0]                 # Planck constant [J-s]
hbar = cst['reduced Planck constant'][0]      # Reduced Planck constant
e = cst['atomic unit of charge'][0]           # Elementary charge [C]
muB = cst['Bohr magneton'][0]                 # Bohr Magneton
g = -cst['electron g factor'][0]              # G-factor
kB = cst['Boltzmann constant'][0]             # Boltzmann constant [J/K]

#Sublattice parameters
Na = 8.441e27             # Moments per unit volume in sublattice A
Nb = 12.66e27             # Moments per unit volume in sublattice B
Ja = 5/2                  # Total AM quantum number for sublattice A
Jb = 5/2                  # Total AM quantum number for sublattice B
ga = g                    # Lande g-factor for sublattice A  
gb = g                    # Lande g-factor for sublattice B

mua_max = ga*muB*Ja       # Maximum moment on sublattice A
Ma_max = Na*ga*muB*Ja     # Maximum magnetization of sublattice A
mub_max = gb* muB*Jb      # Maximum moment on sublattice B
Mb_max = Nb*gb*muB*Jb     # Maximum magnetization of sublattice B

### Function definitions
###______________________________________________________________

def coth(x):
    return 1/np.tanh(x)


def brillouin(y, J):
    eps = 1e-3 # should be small
    y = np.array(y); B = np.empty(y.shape)
    m = np.abs(y)>=eps # mask for selecting elements 
                       # y[m] is data where |y|>=eps;
                       # y[~m] is data where |y|<eps;
    
    B[m] = (2*J+1)/(2*J)*coth((2*J+1)*y[m]/(2*J)) - coth(y[m]/(2*J))/(2*J)
    
    # First order approximation for small |y|<eps
    # Approximation avoids divergence at origin
    B[~m] = ((2*J+1)**2/J**2/12-1/J**2/12)*y[~m]
    
    return B


    def mag_eq_a(Mb, lam_ab, T):
    arg = -mu0 * mua_max * lam_ab * Mb / (kB*T)
    return Ma_max * brillouin(arg, Ja)


def mag_eq_b(Ma, lam_ab, T):
    arg = -mu0 * mua_max * lam_ab * Ma / (kB*T)
    return Mb_max * brillouin(arg, Jb)


def equations(mags, lam_ab, T):
    Ma, Mb = mags
    eq1 = mag_eq_a(Mb, lam_ab, T) - Ma
    eq2 = mag_eq_b(Ma, lam_ab, T) - Mb
    return (eq1, eq2)


def get_mag(T_min, T_max, numpoints, lam_ab):
    Tvec = np.linspace(T_min, T_max, numpoints)
    Ma = np.empty(numpoints)
    Mb = np.empty(numpoints)
    guess = [-Mb_max, Ma_max] # Initial guess
    
    for i in range(numpoints):
        mb, ma = fsolve(equations, x0=guess, args=(lam_ab, Tvec[i]))
        Ma[i] = ma; Mb[i] = mb # Update solution
        guess = [mb, ma]       # Update guess to last solution
        
    return (Tvec, Ma, Mb)


    ### Sliders and buttons
###______________________________________________________________

# Coupling constants
lam_ab_loc = plt.axes([0.125, 0.15, 0.775, 0.03])
lam_ab_init = 500.
lam_ab_max = 1000.
lam_ab_min = 0.
lam_ab_sl = Slider(lam_ab_loc, label=r'$\lambda_{ab}$', valmin=lam_ab_min, \
                   valmax=lam_ab_max, valinit=lam_ab_init)
lam_ab_sl.label.set_size(16)

# Temperature
T_loc = plt.axes([0.125, 0.10, 0.775, 0.03])
T_init = 300.
T_max = 600.
T_min = 1.
T_sl = Slider(T_loc, label=r'$T$ (K)', valmin=T_min, valmax=T_max, \
              valinit=T_init)
T_sl.label.set_size(16)

# Reset button
rst_loc = plt.axes([0.125, 0.9, 0.15, 0.07])
rst_button = Button(rst_loc, 'Reset Sliders', color='C4', hovercolor='C3')
rst_button.label.set_size(16)

