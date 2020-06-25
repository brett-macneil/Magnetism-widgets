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