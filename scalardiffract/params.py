##########################################################################
# Copyright (c) 2023 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides functions to initialize diffraction parameters.
#
##########################################################################

import math
import numpy as np


def grid(pitch, pixels):

    """ Return centered x and y coordinate grids with given pitch and
    pixel number. Pitch and pixel number may either be given as single
    number for both axes or as (x,y) tuple. The first index of the
    returned 2D arrays addresses y and the second index x coordinates.
    """

    # Initialize pixel pitches
    try:
        px, py = pitch
    except:
        px = py = float(pitch)

    # Initialize pixel numbers
    try:
        nx, ny = pixels
    except:
        nx = ny = int(pixels)

    # Initialize coordinate values
    x = (np.arange(nx) - 0.5*(nx-1)) * px
    y = (np.arange(ny) - 0.5*(ny-1)) * py

    # Generate grid arrays
    x, y = np.meshgrid(x, y, indexing="xy")
    return x, y


def max_size(z, p_src):

    """ Return maximum image size based on given image distance and
    pixel pitch of the source. The source pitch may be given as single
    number for both axes or as (x, y) tuple. The size is returned as
    single value or (x,y) tuple acordingly. """

    # Maximum image size as tuple
    try:
        px_src, py_src = p_src
        sx_img = 2*z * math.tan(math.asin(1.0/(2*px_src)))
        sy_img = 2*z * math.tan(math.asin(1.0/(2*py_src)))
        s_img = (sx_img, sy_img)

    # Maximum image size as single value
    except:
        s_img = 2*z * math.tan(math.asin(1.0/(2*p_src)))

    # Return maximum image size
    return s_img


def min_pitch(z, p_src, n_src):

    """ Return minimum image pixel pitch based on given image distance
    and pixel pitch and pixel number of the source. Both, the source
    pitch and the pixel number may be given as single number for both
    axes or as (x, y) tuple. The minimum image pixel pitch is returned
    as single value, if source pitch and pixel number are given as
    single values and as (x,y) tuple otherwise. """

    # Default is a single pitch value
    single = True
    
    # Initialize pixel pitches
    try:
        px_src, py_src = p_src
        single = False
    except:
        px_src = py_src = float(p_src)

    # Initialize pixel numbers
    try:
        nx_src, ny_src = n_src
        single = False
    except:
        nx_src = ny_src = int(n_src)

    # Minimum x pixel pitch
    px_img = z * np.tan(np.arcsin(1.0/(nx_src*px_src)))
    if single:
        return px_img

    # Minimum y pixel pitch    
    py_img = z * np.tan(np.arcsin(1.0/(ny_src*py_src)))

    # Return tuple of minimum pixel pitches
    return px_img, py_img


def opt_params(z, p_src, n_src, over=None, fraction=None, s_img=None):

    """ Return optimum image pixel pitch and image pixel number based on
    given image distance and pixel pitch and pixel number of the source.
    Both, the source pitch and the pixel number may be given as single
    number for both axes or as (x, y) tuple. The return values are given
    as single values, if source pitch and pixel number are provided as
    single values and as (x,y) tuples otherwise.

    An oversampling value (or tuple) may be given to reduce the pixel
    pitch. A image fraction (or tuple) may be used to limit the image
    size relative to the maximum image size based on the source pixel
    pitch. Default is 1. A desired image size (single value or tuple)
    may be given alternatively. Specifying the image size supersedes the
    any fraction value."""
    
    # Default is a single pitch value
    single = True
    
    # Initialize pixel pitches
    try:
        px_src, py_src = p_src
        single = False
    except:
        px_src = py_src = float(p_src)

    # Initialize pixel numbers
    try:
        nx_src, ny_src = n_src
        single = False
    except:
        nx_src = ny_src = int(n_src)

    # Maximum image size
    sx_img, sy_img = max_size(z, (px_src, py_src))

    # Fixed image size
    if s_img is not None:
        try:
            sfx_img, sfy_img = s_img
            single = False
        except:
            sfx_img = sfy_img = float(s_img)
        if sfx_img > sx_img or sfy_img > sy_img:
            raise RuntimeError("Image size too large!")
        fx_img = sfx_img/sx_img
        fy_img = sfy_img/sy_img
        sx_img, sy_img = sfx_img, sfy_img

    # Reduce image size
    elif fraction is not None:
        try:
            fx_img, fy_img = fraction
            single = False
        except:
            fx_img = fy_img = float(fraction)
        if fx_img < 0.0 or fx_img > 1.0 or fy_img < 0.0 or fy_img > 1.0:
            raise RuntimeError("Wrong image fraction!")
        sx_img *= fx_img
        sy_img *= fy_img

    # Minimum image pixel pitch
    px_img, py_img = min_pitch(z, (px_src, py_src), (nx_src, ny_src))

    # Image pixel numbers    
    nx_img = int(round((over * sx_img / px_img), 0))
    ny_img = int(round((over * sy_img / py_img), 0))

    # Adjust image pixel pitches
    px_img = sx_img / nx_img
    py_img = sy_img / ny_img

    # Return image pixel pitch and image pixel number
    if single:
        p_img = px_img
        n_img = nx_img
        fraction = fx_img
    else:
        p_img = (px_img, py_img)
        n_img = (nx_img, ny_img)
        fraction = (fx_img, fy_img)
    return p_img, n_img, fraction


