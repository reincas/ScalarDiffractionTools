##########################################################################
# Copyright (c) 2023 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides beam related functions.
#
##########################################################################

import numpy as np
from skimage.restoration import unwrap_phase


def _beam_width(u, r, ratio="1/e"):

    """ Return beam width of given field magnitude and coordinate
    vectors. The focus width is the full width at the given ratio of the
    field magnitude. Return None, if the detection fails. """

    # Make ratio a number
    if ratio == "1/e":
        ratio = 1.0 / np.e

    # Turn half maximum crossings to zero crossings
    s = u / np.max(u) - ratio

    # Determine zero crossing locations
    i = np.argwhere(s[1:]*s[:-1] < 0)
    i = np.array(i.flat)
    if len(i) < 2:
        return None
    #n = len(u)
    #j = bisect.bisect_left(i, n//2)
    #i = i[j-1:j+1]
    #i = np.array([i[0],i[-1]])
    i = i[[0,-1]]

    # Linear interpolation to refine the zero crossing locations
    s1, s2 = s[i], s[i+1]
    if s1[0] > 0.0:
        return None
    r1, r2 = r[i], r[i+1]
    r0 = r1 - s1*(r2-r1)/(s2-s1)

    # Return the focus width
    return r0[1] - r0[0]


def beam_width(u, x, y, ratio="1/e"):

    """ Return beam width of given complex field and coordinate grids.
    The beam width is the full width at the given ratio of the field
    magnitude in the center of the field array. Return the width in x
    and y direction. Width is None, if the detection fails in the
    respective direction. """

    # Make ratio a number
    if ratio == "1/e":
        ratio = 1.0 / np.e

    # Size of the field grid
    ny, nx = u.shape

    # Width in x direction
    ux = np.abs(u)[nx//2,:]
    x = x[nx//2,:]
    wx = _beam_width(ux, x, ratio)

    # Width in y direction
    uy = np.abs(u)[:,ny//2]
    y = y[:,ny//2]
    wy = _beam_width(uy, y, ratio)

    # Return width in x and y direction
    return wx, wy
    

def _cross(x1, y1, z1, x2, y2, z2):

    """ Return cross product of the given vectors. """
    
    x3 = y1*z2 - z1*y2
    y3 = z1*x2 - x1*z2
    z3 = x1*y2 - y1*x2
    return x3, y3, z3


def get_focus(u, x, y):

    """ Determine the local phase front normal in the center of each
    quadratic tile of coordination pixels. Take a ray through the center
    point in the direction of the normal vector. Find its closest point
    to the optical axis. Return the z position of the closest point as
    well as its distance to the optical axis. """

    # Corner points of the quadratic coodinate matrix tiles
    x0 = x3 = x[:-1,:-1]
    x1 = x2 = x[1:,1:]
    y0 = y1 = y[:-1,:-1]
    y2 = y3 = y[1:,1:]

    # Get the phase shift of the local field
    z = -unwrap_phase(np.angle(u)) / (2*np.pi)
    z0 = z[:-1,:-1]
    z1 = z[:-1,1:]
    z2 = z[1:,1:]
    z3 = z[1:,:-1]

    # Normal vector of the v1-v0 and the v3-v0 vector as well as of the
    # v3-v2 and v1-v2 vector. Take mean of these slithly different
    # vectors as approximation of the phase front normal in the center
    # of the quadratic tiles.
    nx1, ny1, nz1 = _cross(x1-x0, y1-y0, z1-z0, x3-x0, y3-y0, z3-z0) 
    nx2, ny2, nz2 = _cross(x3-x2, y3-y2, z3-z2, x1-x2, y1-y2, z1-z2)
    nx = (nx1 + nx2) / 2
    ny = (ny1 + ny2) / 2
    nz = (nz1 + nz2) / 2

    # Center of the tiles
    x = (x0 + x1) / 2
    y = (y0 + y3) / 2

    # Take a ray through the center of each tile in the direction of the
    # normal vector. Determine closest point to the optical axis and
    # calculate its z position and axis distance.
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -(nx*x + ny*y) / (nx*nx + ny*ny)
    zc = s*nz
    rc = np.sqrt((x + s*nx)**2 + (y + s*ny)**2)
    
    # Return z position and and distance of the closest point
    return zc, rc


