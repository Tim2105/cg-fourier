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

#include <fstream>
#include <iostream>
#include <vector>

#include "helper.h"

//== CLASS DEFINITION =========================================================

/// class that performs 1D fourier transform and different filters
class Fourier1D
{
public:
    /// load signal from file `filename`
    void load_signal(const std::string &filename);

    /// set member `signal_` to `signal`
    void set_signal(const Signal &signal);

    /// compute 1D FT from `signal_` to `spectrum_`
    void fourier_transform();

    /// compute inverse 1D FT from `filtered_spectrum_` to `filtered_signal_`
    void inverse_fourier_transform();

    /// return signal's x-values [0, N)
    Signal &get_signal_x();

    /// return signal's y-values
    Signal &get_signal_y(bool filtered = false);

    /// return spectrums's x-values [-N/2, N/2)
    Signal &get_spectrum_x();

    /// return spectrums's absolut y-values
    Signal &get_spectrum_y();

    /// set `filtered_spectrum_` to `spectrum_` (undo filter)
    void reset_filtered();

    /// set all frequencys above `frequency_k` to zero
    float frequency_filter(int frequency_k, bool filter_above);

    /// set all frequencys below `frequency_k` to zero
    float high_pass_filter(int frequency_k);

    /// set all frequencys with absolut values above `Fk` to zero
    float frequency_value_filter(double Fk, bool filter_above);

    /// set all frequencys with absolut values below `Fk` to zero
    float filter_below(double Fk);

    /// add the real value `add` to the real value corresponding to `frequency_k`
    void add_to_frequency_value(int frequency_k, double add);

    /// compute the absolute y-values used to plot the spectrum
    void compute_spectrum_y(bool filtered = false);

private:
    /// the original (input) signal
    Signal signal_;

    /// the spectrum corresponding to `signal_`
    Spectrum spectrum_;

    /// the `spectrum_` after applying a filter
    Spectrum filtered_spectrum_;

    /// the signal corresponding to the `filtered_spectrum_`
    Signal filtered_signal_;

    /// the x-values used to plot the signal
    Signal signal_x_;

    /// the x-values used to plot the spectrum
    Signal spectrum_x_;

    /// the y-values used to plot the spectrum
    Signal spectrum_y_;
};

//=============================================================================
