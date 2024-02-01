//=============================================================================
//
//   Exercise code for the lecture
//   "Computer Graphics"
//   by Prof. Dr. Mario Botsch, TU Dortmund
//
//   Copyright (C) Computer Graphics Group, TU Dortmund.
//
//=============================================================================

#include "fourier1D.h"

//=============================================================================

void Fourier1D::load_signal(const std::string &filename)
{
    signal_.clear();
    std::ifstream ifs(filename);
    if (!ifs)
    {
        std::cerr << "Cannot read " << filename << std::endl;
        return;
    }

    // read data
    float x;
    while (ifs >> x)
    {
        signal_.push_back(x);
    }

    ifs.close();

    std::cout << "Successfully loaded signal from: " << filename << std::endl;

    // directly apply transforms to signal
    fourier_transform();
    inverse_fourier_transform();
}

//-----------------------------------------------------------------------------

void Fourier1D::set_signal(const Signal &signal)
{
    signal_ = signal;

    // directly apply transforms to signal
    fourier_transform();
    inverse_fourier_transform();
}

//-----------------------------------------------------------------------------


void Fourier1D::fourier_transform()
{
    Signal &f_in = signal_;
    Spectrum &F_out = spectrum_;

    const int N = f_in.size();
    F_out.clear();
    F_out.resize(N, complex(0));

    constexpr double PI = 3.141592654;

#pragma omp parallel for
    for (int k = 0; k < N; ++k)
    {
        double kShifted = k - N / 2;

        complex F_k(0);
        for(int n = 0; n < N; n++) {
            F_k += f_in[n] * exp_i(-2.0 * PI * (double)kShifted * (double)n / (double)N);
        }

        F_k /= (double)N;

        F_out[k] = F_k;
    }

    // initialize filtered spectrum with spectrum (no filter)
    filtered_spectrum_ = spectrum_;
    compute_spectrum_y();
}

//-----------------------------------------------------------------------------

void Fourier1D::inverse_fourier_transform()
{
    Spectrum &F_in = filtered_spectrum_;
    Signal &f_out = filtered_signal_;

    const int N = F_in.size();
    f_out.clear();
    f_out.resize(N, 0.0);

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        complex f_k(0.0);

        for(int k = -N/2; k <= N/2; k++) {
            f_k += F_in[k] * exp_i(2.0 * 3.141592654 * (double)i / (double)N);
        }

        f_out[i] = f_k.real();
    }
}

//-----------------------------------------------------------------------------

float Fourier1D::frequency_filter(int frequency_k, bool filter_above)
{
    const Spectrum &F_in = spectrum_;
    Spectrum &F_out = filtered_spectrum_;
    float percetage_filtered(0.0f);

    const int N = F_in.size();
    F_out = F_in;

  if(filter_above) {
    for(int i = frequency_k + 1; i < F_out.size() / 2; i++) {
        F_out[F_out.size() / 2 - i] = 0;
        F_out[i + F_out.size() / 2] = 0;
        percetage_filtered += 2.0;
    }
  } else {
    for(int i = 0; i < frequency_k; i++) {
        F_out[F_out.size() / 2 - i] = 0;
        F_out[i + F_out.size() / 2] = 0;
        percetage_filtered += 2.0;
    }
  }


    compute_spectrum_y(true);

    return percetage_filtered / F_out.size();
}

//-----------------------------------------------------------------------------

float Fourier1D::frequency_value_filter(double Fk, bool filter_above)
{
    const Spectrum &F_in = spectrum_;
    Spectrum &F_out = filtered_spectrum_;
    float percetage_filtered(0.0f);

    const int N = F_in.size();
    F_out = F_in;

  if(filter_above) {
    for(int i = 0; i < F_out.size(); i++) {
        if(std::abs(F_out[i]) > Fk) {
            F_out[i] = 0;
            percetage_filtered += 1.0;
        }
    }
  } else {
    for(int i = 0; i < F_out.size(); i++) {
        if(std::abs(F_out[i]) < Fk) {
            F_out[i] = 0;
            percetage_filtered += 1.0;
        }
    }
  }


    compute_spectrum_y(true);

    return percetage_filtered / F_out.size();
}

//-----------------------------------------------------------------------------

void Fourier1D::add_to_frequency_value(int frequency_k, double add)
{
    const Spectrum &F_in = spectrum_;
    Spectrum &F_out = filtered_spectrum_;
    const int N = F_in.size();

    F_out[N / 2 + frequency_k] = F_in[N / 2 + frequency_k] + add;
    F_out[N / 2 - frequency_k] = F_in[N / 2 - frequency_k] + add;

    compute_spectrum_y(true);
}

//-----------------------------------------------------------------------------

void Fourier1D::compute_spectrum_y(bool filtered)
{
    Spectrum &F = filtered ? filtered_spectrum_ : spectrum_;
    const int N = signal_.size();
    spectrum_y_.resize(N);

    for (int i = 0; i < N; i++)
        spectrum_y_[i] = abs(F[i]);
}

//-----------------------------------------------------------------------------

Signal &Fourier1D::get_signal_x()
{
    // construct signal x if not done already
    if (signal_x_.size() != signal_.size())
    {
        const int N = signal_.size();
        signal_x_.resize(N);
        for (int i = 0; i < N; i++)
            signal_x_[i] = i / static_cast<float>(N);
    }

    return signal_x_;
}

//-----------------------------------------------------------------------------

Signal &Fourier1D::get_signal_y(bool filtered)
{
    return filtered ? filtered_signal_ : signal_;
}

//-----------------------------------------------------------------------------

Signal &Fourier1D::get_spectrum_x()
{
    // construct spectrum x if not done already
    if (spectrum_x_.size() != signal_.size())
    {
        const int N = signal_.size();
        spectrum_x_.resize(N);
        for (int i = 0; i < N; i++)
            spectrum_x_[i] = i - N / 2;
    }

    return spectrum_x_;
}

//-----------------------------------------------------------------------------

Signal &Fourier1D::get_spectrum_y()
{
    if (spectrum_y_.size() != signal_.size())
    {
        spectrum_y_.resize(signal_.size(), 0.0);
    }

    return spectrum_y_;
}

//-----------------------------------------------------------------------------

void Fourier1D::reset_filtered()
{
    filtered_spectrum_ = spectrum_;
    compute_spectrum_y(true);
    inverse_fourier_transform();
}

//=============================================================================
