##########################################################################
# Copyright (c) 2023 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides functions for the initialization and modification
# of complex fields.
#
##########################################################################

import numpy as np


def plain_wave(x, y=None):

    """ Return a complex plain wave field on the given coordinatze grid.
    If the y coordinates are given, the shape of both arrays must be the
    same. """

    if y is not None:
        if not y.shape == x.shape:
            raise RuntimeError("Grid sizes do mot match!")
        
    return np.ones(x.shape, dtype=complex)


def gaussian_wave(z, w0, x, y):

    """ Return a centered complex field array of Gaussian beam with
    given radius w0 in distance z. """

    r2 = x**2 + y**2
    zr = np.pi * w0**2
    w = w0 * np.sqrt(1 + (z/zr)**2)
    R = z * (1 + (zr/z)**2)
    gouy = np.arctan2(z, zr)

    arg = -r2/w**2 - 2j*np.pi * (r2/(2*R) + z - gouy/(2*np.pi))
    return w0/w * np.exp(arg)


def spherical_wave(f, x, y):

    """ Return complex phase factor of a spherical wave with given focal
    length on the given coordinate grid. All positions are multiples of
    the wavelength. The sign of f matches the sign of the respective z
    position of the focus. """

    return np.exp(2j*np.pi* f * (1 - np.sqrt(1 + (x/f)**2 + (y/f)**2)))


def circle_mask(r, x, y):

    """ Return an array with value 1.0 inside a circle of radius r
    centered at the x,y origin and 0.0 outside. """
    
    return np.where(x**2 + y**2 > r**2, 0.0, 1.0)


def wave_power(u, pitch):

    """ Return power of the complex field weighted with given pitch of
    the coordinate grid. The pitch may either be a single value for both
    axes or a (x,y) tuple. """

    # Initialize pixel pitches
    try:
        px, py = pitch
    except:
        px = py = float(pitch)

    # Return power value
    return np.sum(np.abs(u)**2 * px*py)


