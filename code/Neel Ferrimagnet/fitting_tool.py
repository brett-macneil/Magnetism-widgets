#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2020
Last updated Mon Jul 6 2020

@author: Brett MacNeil
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, Button
from scipy.optimize import fsolve, curve_fit
from scipy.constants import physical_constants as cst

### Constants
###______________________________________________________________

filename = ''

numpoints = 70           # Number of points used in equation solver

T_max = 600
T_min = 1
T_vec = np.linspace(T_min, T_max, numpoints)

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
    
    B[m] = (2*J+1)/(2*J)*coth((2*J+1)*y[m]/(2*J)) - coth(y[m]/(2*J))/(2*J)
    
    # First order approximation for small |y|<eps
    # Approximation avoids divergence at origin
    B[~m] = ((2*J+1)**2/J**2/12-1/J**2/12)*y[~m]
    
    return B


def mag_eq_a(Ma, Mb, lam_aa, lam_ab, T):
    arg = mu0 * mua_max * (- lam_aa * Ma - lam_ab * Mb) / (kB*T)
    return Ma_max * brillouin(arg, Ja)


def mag_eq_b(Ma, Mb, lam_bb, lam_ab, T):
    arg = mu0 * mua_max * (- lam_ab * Ma - lam_bb * Mb) / (kB*T)
    return Mb_max * brillouin(arg, Jb)


def equations(mags, lam, T):
    Ma, Mb = mags
    lam_aa, lam_bb, lam_ab, lam_ba = lam
    eq1 = mag_eq_a(Ma, Mb, lam_aa, lam_ab, T) - Ma
    eq2 = mag_eq_b(Ma, Mb, lam_bb, lam_ba, T) - Mb
    return (eq1, eq2)


def get_mag(T, lam_aa, lam_bb, lam_ab, lam_ba):
    lam = [lam_aa, lam_bb, lam_ab, lam_ba]
    Ma = np.empty(numpoints)
    Mb = np.empty(numpoints)
    guess = [-Ma_max, Mb_max] # Initial guess
    
    for i in range(numpoints):
        ma, mb = fsolve(equations, x0=guess, args=(lam, T[i]))
        Ma[i] = ma; Mb[i] = mb
        guess = [ma, mb]
        
    return Ma+Mb


### Set up figure
###______________________________________________________________

fig, ax = plt.subplots(1, 1)
plt.subplots_adjust(bottom=0.35, left=0.15)

ax.set_xlabel('Temperature (K)', fontsize=16)
ax.set_ylabel(r'Magnetization (kA m$^{-1}$)', fontsize=16)
ax.grid(color='grey', linestyle='dotted', linewidth=1)


### Sliders and buttons
###______________________________________________________________

# Coupling constants
lam_aa_loc = plt.axes([0.125, 0.20, 0.775, 0.03])
lam_aa_init = 0.
lam_aa_max = 1000.
lam_aa_min = 0.
lam_aa_sl = Slider(lam_aa_loc, label=r'$\lambda_{aa}$', valmin=lam_aa_min, \
                   valmax=lam_aa_max, valinit=lam_aa_init)
lam_aa_sl.label.set_size(16)

lam_bb_loc = plt.axes([0.125, 0.15, 0.775, 0.03])
lam_bb_init = 0.
lam_bb_max = 1000.
lam_bb_min = 0.
lam_bb_sl = Slider(lam_bb_loc, label=r'$\lambda_{bb}$', valmin=lam_bb_min, \
                   valmax=lam_bb_max, valinit=lam_bb_init)
lam_bb_sl.label.set_size(16)

lam_ab_loc = plt.axes([0.125, 0.10, 0.775, 0.03])
lam_ab_init = 500.
lam_ab_max = 1000.
lam_ab_min = 0.
lam_ab_sl = Slider(lam_ab_loc, label=r'$\lambda_{ab}$', valmin=lam_ab_min, \
                   valmax=lam_ab_max, valinit=lam_ab_init)
lam_ab_sl.label.set_size(16)

lam_ba_loc = plt.axes([0.125, 0.05, 0.775, 0.03])
lam_ba_init = 500.
lam_ba_max = 1000.
lam_ba_min = 0.
lam_ba_sl = Slider(lam_ba_loc, label=r'$\lambda_{ba}$', valmin=lam_ba_min, \
                   valmax=lam_ba_max, valinit=lam_ba_init)
lam_ba_sl.label.set_size(16)

# Reset button
rst_loc = plt.axes([0.15, 0.9, 0.25, 0.07])
rst_button = Button(rst_loc, 'Reset Sliders', color='C4', hovercolor='C3')
rst_button.label.set_size(16)

# Refine button
ref_loc = plt.axes([0.6, 0.9, 0.30, 0.07])
ref_button = Button(ref_loc, 'Refine Solution', color='C4', hovercolor='C3')
ref_button.label.set_size(16)


### Plotting
###______________________________________________________________

T, M = np.genfromtxt(filename, unpack=True)

ax.plot(T, M/1e3, marker='.', ls='')

magplot, = ax.plot(T_vec, get_mag(T_vec, lam_aa_init, lam_bb_init, \
                                  lam_ab_init, lam_ba_init)/1e3)
    

### Updates
###______________________________________________________________

def update(val):
    magplot.set_ydata(get_mag(T_vec, lam_aa_sl.val, lam_bb_sl.val, \
                              lam_ab_sl.val, lam_ba_sl.val)/1e3)
    return None


def reset(event):
    lam_aa_sl.reset()
    lam_bb_sl.reset()
    lam_ab_sl.reset()
    lam_ba_sl.reset()
    return None

    
def refine(event):
    guess = [lam_aa_sl.val, lam_bb_sl.val, lam_ab_sl.val, lam_ba_sl.val]
    print('\n Original parameters:', guess, '\n')
    
    lam_ref = curve_fit(get_mag, xdata=T, ydata=M, p0=guess)[0]
    print('\n Refined parameters:', lam_ref, '\n')
    
    M_ref = get_mag(T_vec, lam_ref[0], lam_ref[1], lam_ref[2], lam_ref[3])/1e3
    magplot.set_ydata(M_ref)
    
    return None


lam_aa_sl.on_changed(update)
lam_bb_sl.on_changed(update)
lam_ab_sl.on_changed(update)
lam_ba_sl.on_changed(update)

rst_button.on_clicked(reset)
ref_button.on_clicked(refine)