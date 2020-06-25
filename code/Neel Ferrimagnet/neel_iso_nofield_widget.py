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