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

#include "data2D.h"

//== CLASS DEFINITION =========================================================

class Fourier2D
{
public:
    /// enum for different types of filters that are applied to the spectrum
    enum class FilterType
    {
        None,
        Gaussian,
        Inverse_Gaussian
    };

    /// constructor
    Fourier2D()
        : filter_type_(FilterType::None), kernel_width_(10), colored_(false)
    {
    }

    /// load signal from file `filename`
    bool load_signal(const std::string &filename);

    /// wrapper for both colored and greyscale FT
    void FT();

    /// wrapper for both colored and greyscale inverse FT
    void iFT();

    /// wrapper for different filter methods
    void filter();

    /// compute a hybrid image from `signal_` and `second_signal`
    void hybrid(Signal2D &second_signal);

    /// `colored` determines if the signals and specra use 1 oder 3 color channels
    void set_colored(bool colored);

private:
    /// the implementation of the 2D FT
    void fourier_transform();

    /// the implementation of the 2D inverse FT
    void inverse_fourier_transform();

    /// filters the spectrum with a gaussian kernel of `kernel_width_`
    void filter_gaussian(bool inverse = false);

public:
    /// the original signal (image) holding color or greyscale values
    Signal2D signal_;

    /// the spectrum corresponding to `signal_`
    Spectrum2D spectrum_;

    /// the result after apllying `filter()` to `spectrum_`
    Spectrum2D filtered_spectrum_;

    /// the signal corresponding to `filtered_spectrum_`
    Signal2D filtered_signal_;

    /// the current filter type that is used in `filter()`
    FilterType filter_type_;

    /// the current kernel width used by the gaussian and inverse gaussian filters
    int kernel_width_;

private:
    /// in case of `colored_`, three color channels will be transformed and
    /// filtered instead of one
    bool colored_;
};

//=============================================================================
