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
from matplotlib.widgets import Slider, Button
from scipy.constants import physical_constants as cst
from scipy.optimize import fsolve, curve_fit


# Load M vs T data to fit
#filename = ''
#T, M = np.genfromtxt(filename, unpack=True)
#M /=1e3 # A/m to kA/m


# Set up figure
fig, ax = plt.subplots(1, 1)
plt.subplots_adjust(bottom=0.35)
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
    y = mu0*mu*lam*M
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


def get_tot_mag(T, lam_1, lam_2, mu_1, mu_2, N1=N, N2=N, kilo=True):
    Jeff_1 = mu_1/(g*muB)
    Jeff_2 = mu_2/(g*muB)
    numpoints = len(T)
    
    Mag_1 = np.empty(numpoints)
    Mag_2 = np.empty(numpoints)
    guess_1 = N1*mu_1
    guess_2 = N2*mu_2
    
    for i in range(numpoints):
        Mag_1[i] = fsolve(mag_eq, x0=guess_1, 
                          args=(lam_1, mu_1, T[i], Jeff_1, N1))
        guess_1 = Mag_1[i]
        Mag_2[i] = fsolve(mag_eq, x0=guess_2, 
                          args=(lam_2, mu_2, T[i], Jeff_2, N2))
        guess_2 = Mag_2[i]
        
    if kilo:
        Mag_1 /= 1e3
        Mag_2 /= 1e3
    
    return Mag_1 - Mag_2


def curie_temp(lam, mu):
    # 1st order expansion of Brillouin 
    # function to estimate Curie temperature.
    Jeff = mu/(g*muB)
    Tc = lam*mu0*N*Jeff*(Jeff+1)*(g*muB)**2
    Tc /= 3*kB
    return Tc


# Sliders and buttons
lam_1_loc = plt.axes([0.125, 0.20, 0.775, 0.03])
lam_1_init = 750.
lam_1_max = 1000.
lam_1_min = 0.
lam_1_sl = Slider(ax=lam_1_loc, label=r'$\lambda_1$', valmin=lam_1_min, 
                  valmax=lam_1_max, valinit=lam_1_init)

lam_2_loc = plt.axes([0.125, 0.15, 0.775, 0.03])
lam_2_init = 750.
lam_2_max = 1000.
lam_2_min = 0.
lam_2_sl = Slider(ax=lam_2_loc, label=r'$\lambda_2$', valmin=lam_2_min, 
                  valmax=lam_2_max, valinit=lam_2_init)

mu_1_loc = plt.axes([0.125, 0.10, 0.775, 0.03])
# Moments in units of muB
mu_1_init = 2.5
mu_1_max = 5.
mu_1_min = 0.
mu_1_sl = Slider(ax=mu_1_loc, label=r'$\mu_1$ $(\mu_B)$', valmin=mu_1_min, 
                  valmax=mu_1_max, valinit=mu_1_init)

mu_2_loc = plt.axes([0.125, 0.05, 0.775, 0.03])
# Moments in units of muB
mu_2_init = 2.5
mu_2_max = 5.
mu_2_min = 0.
mu_2_sl = Slider(ax=mu_2_loc, label=r'$\mu_2$ $(\mu_B)$', valmin=mu_2_min, 
                  valmax=mu_2_max, valinit=mu_2_init)

rst_loc = plt.axes([0.125, 0.9, 0.20, 0.07])
rst_button = Button(rst_loc, 'Reset Sliders', color='C4', hovercolor='C3')

ref_loc = plt.axes([0.7, 0.9, 0.20, 0.07])
ref_button = Button(ref_loc, 'Refine Solution', color='C4', hovercolor='C3')


# Plotting
M1 = get_mag(T, lam_1_init, mu_1_init*muB)
M2 = -get_mag(T, lam_2_init, mu_2_init*muB)
Mtot = M1 + M2

data_plot, = ax.plot(T, M, marker='.', linestyle='', color='C0')
M1_plot, = ax.plot(T, M1, linestyle='dotted', color='C7')
M2_plot, = ax.plot(T, M2, linestyle='dotted', color='C7')
Mtot_plot, = ax.plot(T, Mtot, color='C3')


# Slider updates and button functionality
def update(val):
    lam_1_new = lam_1_sl.val
    lam_2_new = lam_2_sl.val
    mu_1_new = mu_1_sl.val
    mu_1_new *= muB
    mu_2_new = mu_2_sl.val
    mu_2_new *= muB
    
    M1_new = get_mag(T, lam_1_new, mu_1_new)
    M1_plot.set_ydata(M1_new)
    
    M2_new = -get_mag(T, lam_2_new, mu_2_new)
    M2_plot.set_ydata(M2_new)
    
    Mtot_new = M1_new + M2_new
    Mtot_plot.set_ydata(Mtot_new)
    
    fig.canvas.draw_idle()
    
    return None


def reset(event):
    lam_1_sl.reset()
    lam_2_sl.reset()
    mu_1_sl.reset()
    mu_2_sl.reset()
    return None


def refine(event):
    guess = [lam_1_sl.val, lam_2_sl.val, mu_1_sl.val*muB, mu_2_sl.val*muB]
    params_ref = curve_fit(get_tot_mag, xdata=T, ydata=M, p0=guess)[0]
    
    M1_ref = get_mag(T, params_ref[0], params_ref[2])
    M1_plot.set_ydata(M1_ref)
    
    M2_ref = -get_mag(T, params_ref[1], params_ref[3])
    M2_plot.set_ydata(M1_ref)
    
    Mtot_ref = M1_ref + M2_ref
    Mtot_plot.set_ydata(Mtot_ref)
    
    lam_1_sl.set_val(params_ref[0])
    lam_2_sl.set_val(params_ref[1])
    mu_1_sl.set_val(params_ref[2]/muB)
    mu_2_sl.set_val(params_ref[3]/muB)

    
    print('\n','Original parameters','\n',guess,'\n')
    print('\n','Refined parameters','\n',params_ref,'\n')
    
    return None  

    
lam_1_sl.on_changed(update)
lam_2_sl.on_changed(update)
mu_1_sl.on_changed(update)
mu_2_sl.on_changed(update)

rst_button.on_clicked(reset)
ref_button.on_clicked(refine)