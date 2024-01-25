//=============================================================================
//
//   Exercise code for the lecture
//   "Computer Graphics"
//   by Prof. Dr. Mario Botsch, TU Dortmund
//
//   Copyright (C) Computer Graphics Group, TU Dortmund.
//
//=============================================================================

#include "data2D.h"

#include <stb_image.h>
#include <stb_image_write.h>

//=============================================================================

bool Signal2D::load(const std::string &fn)
{
    // load with stb_image
    int width, height, n;
    stbi_set_flip_vertically_on_load(false);
    unsigned char *rgb = stbi_load(fn.c_str(), &width, &height, &n, 3);
    if (!rgb)
    {
        std::cout << "read error! " << fn << std::endl;
        width_ = height_ = 0;
        return false;
    }

    resize(width, height);

    if (num_components_ == 1)
    {
        // convert RGB (char) to grey scale (float)
        for (unsigned int i = 0; i < width_ * height_; ++i)
        {
            data_[i] =
                (rgb[3 * i] + rgb[3 * i + 1] + rgb[3 * i + 2]) / 3.0 / 255.0;
        }
    }
    else
    {
        // convert RGB (char) to RGB (float) and sort by color channel
        for (unsigned int i = 0; i < width_ * height_ * num_components_; ++i)
        {
            data_[(i % 3) * width_ * height_ + i / 3] = rgb[i] / 255.0;
        }
    }

    // free memory
    stbi_image_free(rgb);

    return true;
}

//-----------------------------------------------------------------------------

bool Signal2D::write(const std::string &fn)
{
    // convert float pixel in [0,1] values to unsigned char in [0,255]
    std::vector<unsigned char> rgb(3 * width_ * height_);
    for (unsigned int y = 0; y < height_; ++y)
    {
        for (unsigned int x = 0; x < width_; ++x)
        {
            for (unsigned int c = 0; c < 3; ++c)
            {
                // colored? use rgb channels
                int cc = (num_components_ == 3) ? c : 0;
                double val = data_[y * width_ + x + cc * width_ * height_];
                rgb[3 * (y * width_ + x) + c] = (unsigned char)(std::min(255.0, 255.0 * val));
            }
        }
    }

    // write RGB array to file
    stbi_flip_vertically_on_write(false);
    bool ok =
        stbi_write_png(fn.c_str(), width_, height_, 3, rgb.data(), 3 * width_);
    return ok;
}

//-----------------------------------------------------------------------------

bool Spectrum2D::write(const std::string &fn)
{
    // convert complex pixel values to unsigned char in [0,255]
    std::vector<unsigned char> rgb(3 * width_ * height_);
    for (unsigned int y = 0; y < height_; ++y)
    {
        for (unsigned int x = 0; x < width_; ++x)
        {
            for (unsigned int c = 0; c < 3; ++c)
            {
                // colored? use rgb channels
                int cc = (num_components_ == 3) ? c : 0;
                double val = abs(data_[y * width_ + x + cc * width_ * height_]);

                rgb[3 * (y * width_ + x) + c] =
                    (unsigned char)(255 * std::min(1.0, std::max(0.0, val)));
            }
        }
    }

    // write RGB array to file
    stbi_flip_vertically_on_write(false);
    bool ok =
        stbi_write_png(fn.c_str(), width_, height_, 3, rgb.data(), 3 * width_);
    return ok;
}

//=============================================================================
