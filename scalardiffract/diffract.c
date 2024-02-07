/*************************************************************************
* Copyright (c) 2023 Reinhard Caspary                                    *
* <reinhard.caspary@phoenixd.uni-hannover.de>                            *
* This program is free software under the terms of the MIT license.      *
**************************************************************************
*
* This file provides the Rayleigh-Sommerfeld diffraction formula in the
* function rs_c().
*
*************************************************************************/

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#define PI 3.14159265358979323846

int no_matrix(PyArrayObject *arr) {

	// Parameter must be a 2D matrix
    if (arr->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "Matrix with two dimensions required!");
        return 1;
	}
    return 0;
}

/*
Rayleigh-Sommerfeld diffraction formula: Determine the diffraction field
of a given source field in a certain image distance. This approach is
general with no approximations. With a complexity of O(N^4) it is very
expensive. However, it can serve as a reference to identify the aberrations
caused by faster approaches.

Parameters:
  double z: Image distance in wavelength units
  double px_src: Pixel pitch of the source in wavelength units (x)
  double py_src: Pixel pitch of the source in wavelength units (y)
  double px_img: Pixel pitch of the image in wavelength units (x)
  double py_img: Pixel pitch of the image in wavelength units (y)
  complex *U_src: Matrix of the source field
  complex *U_img: Matrix of the image field

Source and image are centered on the optical axis.
*/

static PyObject* rs_c(PyObject* dummy, PyObject* args)
{
	// Declaration of the function parameter objects
    double z, px_src, py_src, px_img, py_img;
    PyObject* arg_u_src = NULL;
    PyObject* arg_u_img = NULL;
	
	// Retrieve Python objects of function parameters
    if (!PyArg_ParseTuple(args, "dddddOO!", &z, &px_src, &py_src, &px_img, &py_img, &arg_u_src, &PyArray_Type, &arg_u_img))
        return NULL;

	// Obtain Numpy arrays from Python parameter objects
    PyArrayObject *u_src = (PyArrayObject *)PyArray_FROM_OTF(arg_u_src, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *u_img = (PyArrayObject *)PyArray_FROM_OTF(arg_u_img, NPY_CDOUBLE, NPY_ARRAY_OUT_ARRAY);

	// Check validity of the Numpy arrays
    if (u_src == NULL || no_matrix(u_src) || u_img == NULL || no_matrix(u_img)) {
		Py_XDECREF(u_src);
		Py_XDECREF(u_img);		
		return NULL;
	}

	// Pixel numbers of all four dimensions
	int nx_src = (int)u_src->dimensions[1];
	int ny_src = (int)u_src->dimensions[0];
	int nx_img = (int)u_img->dimensions[1];
	int ny_img = (int)u_img->dimensions[0];
	//printf("Source size: (%d, %d)\n", ny_src, nx_src);
	//printf("Image size:  (%d, %d)\n", ny_img, nx_img);

	// Pointers to the C data arrays in the Numpy array structs
	double *u_src_data = (double *)u_src->data;
	double *u_img_data = (double *)u_img->data;

	// Prepare global factor (2*pi)**2 * z * px * py
	double fz = 4*PI*PI * z * px_src*py_src;
	
	// Prepare pixel offset differences
	double dox = 0.5 * ((nx_src - 1) * px_src - (nx_img - 1) * px_img);
	double doy = 0.5 * ((ny_src - 1) * py_src - (ny_img - 1) * py_img);
	
	// Loop though all image pixels k,l
	int l;
	#pragma omp parallel for
    for (l = 0; l < ny_img; l++) {
		for (int k = 0; k < nx_img; k++) {
			
			// Current image data index
			int i_img = 2 * (k + nx_img * l);
			
			// Clear value of current image pixel
			u_img_data[i_img] = 0.0;
			u_img_data[i_img+1] = 0.0;
			
			// Loop through all source pixels i,j
			for (int j = 0; j < ny_src; j++) {
				for (int i = 0; i < nx_src; i++) {
					
					// Current DOE data index
					int i_src = 2 * (i + nx_src * j);
					
					// Prepare RS variables for the fuction g
					double dx = dox + k * px_img - i * px_src;
					double dy = dox + l * py_img - j * py_src;
					double r2 = z*z + dy*dy + dx*dx;
					double p = 2*PI * sqrt(r2);
					double invp2 = 1.0 / (4*PI*PI * r2);
					
					// Magnitude and angle of the function g = (1/p - i) * exp(i*p)/p^2
					double mag = sqrt(1.0 + invp2) * invp2;
					double arg = atan2(-1.0, 1.0/p) + p;
					
					// Magnitude and angle of source field U_src
					mag *= sqrt(u_src_data[i_src]*u_src_data[i_src] + u_src_data[i_src+1]*u_src_data[i_src+1]);
					if (mag != 0.0) {
						arg += atan2(u_src_data[i_src+1], u_src_data[i_src]);
					};

					// Add field of Hygens wave propagating from the current DOE pixel to the current image pixel
					u_img_data[i_img] += mag * cos(arg);
					u_img_data[i_img+1] += mag * sin(arg);
				}
			}
			
			// Multiply image field U_out by global factor (2*pi)**2 * z * p**2
			u_img_data[i_img] *= fz;
			u_img_data[i_img+1] *= fz;			
		}
	}
	
	// Decrement reference counters of numpy arrays
    Py_DECREF(u_src);
    Py_DECREF(u_img);

	//printf("Done.\n");
    //return PyInt_FromLong(3);

	// No return value
    Py_INCREF(Py_None);
    return Py_None;
}

// Method table of the module
static PyMethodDef methods[] = {
        {
                "_rs", rs_c, METH_VARARGS,
                "Rayleigh-Sommerfeld diffraction function",
        },
        {NULL, NULL, 0, NULL}
};

// Module definition structure
static struct PyModuleDef definition = {
        PyModuleDef_HEAD_INIT,
        "_scalardiffract",
        "A Python module for scalar optical diffraction in C code.",
        -1,
        methods
};

// Module initialization function
PyMODINIT_FUNC PyInit__scalardiffract(void) {
	
	// Initialize the Python interpreter to make the C API available
    Py_Initialize();
	
	// Initialize Numpy
    import_array();
	
	// Create cdiffract module
    return PyModule_Create(&definition);
}
