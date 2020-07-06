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