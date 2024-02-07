##########################################################################
# Copyright (c) 2023 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides wave diffraction functions.
#
##########################################################################

import numpy as np
from _scalardiffract import _rs


def rs(z, p_src, p_img, U_src, shape_img):

    """ C version of Rayleigh-Sommerfeld diffraction formula. """

    # Initialize pixel pitch parameters
    try:
        px_src, py_src = p_src
    except:
        px_src = py_src = float(p_src)
    try:
        px_img, py_img = p_img
    except:
        px_img = py_img = float(p_img)
        
    # Prepare quadratic shape
    if isinstance(shape_img, int):
        shape_img = (shape_img, shape_img)
        
    # Create image field matrix based on given shape
    if isinstance(shape_img, tuple):
        if len(shape_img) != 2:
            raise ValueError("Shape must be a 2-tuple")
        U_img = np.zeros(shape_img, dtype=complex)

    # Use given image field matrix
    else:
        U_img = shape_img
        shape_img = U_img.shape

    # Fields must be given as 2D matrices
    if len(U_src.shape) != 2 or len(U_img.shape) != 2:
        raise TypeError("Field must be a matrix!")
        
    # Both field matrices must be of complex type
    if U_src.dtype != complex or U_img.dtype != complex:
        raise TypeError("Field matrix must be complex!")

    # Calculation using C implementation
    _rs(z, px_src, py_src, px_img, py_img, U_src, U_img)

    # Return image field matrix
    return U_img


def py_rs(z, p_src, p_img, U_src, shape_img):

    """ Python version of Rayleigh-Sommerfeld diffraction formula. """

    # Shape must must be 2D
    if len(shape_img) != 2:
        raise ValueError("Shape must be a 2-tuple")

    # Field must be given as 2D matrix
    if len(U_src.shape) != 2:
        raise TypeError("Field must be a matrix!")
        
    # Field matrix must be of complex type
    if U_src.dtype != complex:
        raise TypeError("Field matrix must be complex!")

    # Coordinate vectors of source pixels
    ny_src, nx_src = U_src.shape
    x_src = (np.arange(nx_src) - 0.5*(nx_src-1)) * p_src
    y_src = (np.arange(ny_src) - 0.5*(ny_src-1)) * p_src

    # Coordinate vectors of image pixels
    ny_img, nx_img = shape_img
    x_img = (np.arange(nx_img) - 0.5*(nx_img-1)) * p_img
    y_img = (np.arange(ny_img) - 0.5*(ny_img-1)) * p_img

    # Build 4D coordinate arrays
    y_src, x_src, y_img, x_img = np.meshgrid(y_src, x_src, y_img, x_img,
                                             indexing="ij")

    # Use Rayleigh-Sommerfeld formula to calculate image field
    r = np.sqrt(z**2 + (y_img-y_src)**2 + (x_img-x_src)**2)
    p = 2j * np.pi * r
    g = (1/p - 1) * np.exp(p) / r**2
    U_img = 1j * z * np.tensordot(U_src*w, g, axes=2)

    # Return image field
    return U_img
