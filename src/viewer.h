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

#include <pmp/window.h>

#include "fourier1D.h"
#include "fourier2D.h"

//=============================================================================

// define input directories if something went wrong in cmake
#ifndef DATA_DIRECTORY
#if defined(_WIN32) || defined(__XCODE__)
#define DATA_DIRECTORY ("../../data/")
#else
#define DATA_DIRECTORY ("../data/")
#endif
#endif

using namespace pmp;

//== CLASS DEFINITION =========================================================

/// Exercise viewer
class FourierViewer : public Window
{
public:
    /// constructor
    FourierViewer(const char *title, int width = 0, int height = 0);

    /// destructor
    ~FourierViewer();

private:
    /// mostly empty, usually used to call draw for 3D objects
    virtual void display() override;

    /// draw GUI elements
    virtual void process_imgui() override;

    /// GUI functions for the left 1D FT side bar
    void imgui_1D_controls();

    /// GUI functions for the left 2D FT side bar
    void imgui_2D_controls();

    /// GUI functions for the left 2D filter settings
    void imgui_2D_controls_filter();

    /// GUI functions for the left 2D hybrid settings
    void imgui_2D_controls_hybrid();

    /// show 1D FT plots
    void process_implot_1D();

    /// show 2D FT images
    void process_implot_2D();

    /// show 2D FT filter images
    void process_implot_2D_filter();

    /// show 2D FT hybrid images
    void process_implot_2D_hybrid();

private:
    /// structure holding the relevant variables of an OpenGL texture
    struct ImageTexture
    {
        ImageTexture() : texID(0), width(0), height(0) {}

        GLuint texID;
        int width;
        int height;
    };

    /// create an OpenGL Texture from a 2D signal
    void create_texture_from_2D_signal(const Signal2D &signal,
                                       const bool greyscale,
                                       ImageTexture &out_tex);

    /// create an OpenGL Texture from a 2D spectrum
    void create_texture_from_2D_spectrum(const Spectrum2D &spectrum,
                                         const bool greyscale,
                                         ImageTexture &out_tex);

private:
    /// wrapper object for 1D Fourier Transform
    Fourier1D ft1D_;

    /// wrapper object for 2D Fourier Transform
    Fourier2D ft2D_;

    /// decide if the original 1D signal should be drawn
    bool draw_unfiltered_;

    /// switch between hybrid and filter 2D FT view mode
    bool hybrid_;

    /// switch between colored and greyscale images
    bool colored_;

    /// determine, if some event happened that needs the plots to be resetted to
    /// the original size
    bool rescale_;

    /// determine, if some event happened that needs the original images to read
    /// from file again
    bool reload_;

    /// use nearest image mini- and magnification filter for pixel art
    bool nearest_;

    /// the percentage of 1D spectrum values that were filtered-out
    float percentage_filtered_;

    /// size of the hybrid image texture
    float image_size_;

    /// height of imgui text
    float text_height_;

    /// second signal (image) that is mixed with the one in `ft2D_` in hybrid mode
    Signal2D second_signal_;

    /// OpenGL textures
    ImageTexture tex_signal_;
    ImageTexture tex_spectrum_;
    ImageTexture tex_filtered_spectrum_;
    ImageTexture tex_filtered_signal_;
};

//=============================================================================
