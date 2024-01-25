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

#include <cassert>
#include <iostream>
#include <vector>

#include "helper.h"

//== CLASS DEFINITION =========================================================

/// Template class for 2D data
template <class T>
class Data2D
{
public:
    /// Constructor
    Data2D(size_t w = 0, size_t h = 0)
        : width_(w), height_(h), num_components_(1), component_(0)
    {
        if (w > 0 && h > 0)
            resize(w, h);
    }

    /// resize data (destroys its content)
    void resize(size_t w, size_t h)
    {
        width_ = w;
        height_ = h;
        data_.clear();
        data_.resize(width_ * height_ * num_components_);
    }

    /// copy data array from another Data2D class object
    void copy_data(const Data2D &other)
    {
#pragma omp parallel for
        for (int i = 0; i < data_.size(); i++)
        {
            data_[i] = other.data()[i];
        }
    }

    /// abstract function to write the data to a 2D image
    virtual bool write(const std::string &fn) = 0;

    /// get (read access)
    T operator()(size_t _x, size_t _y) const
    {
        assert(_x < width_);
        assert(_y < height_);

        return data_[_y * width_ + _x + component_ * width_ * height_];
    }

    /// get (read/write access)
    T &operator()(size_t _x, size_t _y)
    {
        assert(_x < width_);
        assert(_y < height_);

        return data_[_y * width_ + _x + component_ * width_ * height_];
    }

    /// get data array/pointer
    const T *data() const { return data_.data(); }

    /// set the current component used for operator() (0 = red, 1 = green, 2 =
    /// blue)
    void set_component(size_t component) { component_ = component; }

    /// set the max number of components (1 = grey, 3 = rgb)
    void set_num_components(size_t num_components)
    {
        num_components_ = num_components;
        component_ = 0;
        resize(width_, height_);
    }

    /// horizontal and vertical size of the data array
    size_t width_, height_;

protected:
    /// linear array holding the data values
    std::vector<T> data_;

    /// number of components/color channels (1 = grey, 3 = rgb)
    size_t num_components_;

    /// the current component used for read/wirte access to data via operator()
    size_t component_;
};

//== CLASS DEFINITION =========================================================

/// derived class holding double values (2D images)
class Signal2D : public Data2D<double>
{
public:
    /// constructor
    Signal2D(size_t w = 0, size_t h = 0) : Data2D(w, h) {}

    /// read data from an image file (`fn`) to the data array
    bool load(const std::string &fn);

    /// write data to an image file (`fn`)
    bool write(const std::string &fn) override final;
};

//== CLASS DEFINITION =========================================================

/// derived class holding complex values (2D spectrums)
class Spectrum2D : public Data2D<complex>
{
public:
    /// constructor
    Spectrum2D(size_t w = 0, size_t h = 0) : Data2D(w, h) {}

    /// write abs(data) to an image file (`fn`)
    bool write(const std::string &fn) override final;
};

//=============================================================================
