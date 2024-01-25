//=============================================================================
//
//   Exercise code for the lecture
//   "Computer Graphics"
//   by Prof. Dr. Mario Botsch, TU Dortmund
//
//   Copyright (C) Computer Graphics Group, TU Dortmund.
//
//=============================================================================
#pragma once
//=============================================================================

//== INCLUDES =================================================================

#include <complex>

//=============================================================================

// our complex number type
typedef std::complex<double> complex;

// the signal is stored in a vector of float
typedef std::vector<double> Signal;

// the frequency spectrum is stored in a vector of complex
typedef std::vector<complex> Spectrum;

//-----------------------------------------------------------------------------

inline complex exp_i(double x)
{
    return complex(cos(x), sin(x));
}
