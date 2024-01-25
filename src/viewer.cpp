//=============================================================================
//
//   Exercise code for the lecture
//   "Computer Graphics"
//   by Prof. Dr. Mario Botsch, TU Dortmund
//
//   Copyright (C) Computer Graphics Group, TU Dortmund.
//
//=============================================================================

#include "viewer.h"

#include <fstream>
#include <imgui.h>
#include <implot.h>
#include <stb_image.h>

//=============================================================================

FourierViewer::FourierViewer(const char *_title, int _width, int _height)
    : Window(_title, _width, _height),
      draw_unfiltered_(true),
      hybrid_(false),
      colored_(false),
      rescale_(false),
      reload_(false),
      nearest_(false),
      percentage_filtered_(0.0f),
      image_size_(1.0)
{
    // init implot
    ImPlot::CreateContext();

    // define some ImGui styles that are not already defined in window base class
    ImGui::GetStyle().Colors[ImGuiCol_Tab] = ImVec4(0.16f, 0.62f, 0.87f, 0.40f);
    ImGui::GetStyle().Colors[ImGuiCol_TabActive] =
        ImVec4(0.16f, 0.62f, 0.87f, 1.00f);
    ImGui::GetStyle().Colors[ImGuiCol_TabHovered] =
        ImVec4(0.16f, 0.62f, 0.87f, 1.00f);

    // set clear color to white
    glClearColor(1.0, 1.0, 1.0, 1.0);
}

//-----------------------------------------------------------------------------

FourierViewer::~FourierViewer()
{
    ImPlot::DestroyContext();

    glDeleteTextures(1, &tex_signal_.texID);
    glDeleteTextures(1, &tex_spectrum_.texID);
    glDeleteTextures(1, &tex_filtered_signal_.texID);
    glDeleteTextures(1, &tex_filtered_spectrum_.texID);
}

//-----------------------------------------------------------------------------

void FourierViewer::display()
{
    // clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

//-----------------------------------------------------------------------------

void FourierViewer::process_imgui()
{
    Window::process_imgui();

    if (ImGui::BeginTabBar("FourierTabs"))
    {
        if (ImGui::BeginTabItem("1D Fourier"))
        {
            ImGui::Spacing();
            imgui_1D_controls();
            process_implot_1D();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("2D Fourier"))
        {
            ImGui::Spacing();
            imgui_2D_controls();
            process_implot_2D();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

//-----------------------------------------------------------------------------

void FourierViewer::imgui_1D_controls()
{
    if (ImGui::CollapsingHeader("Signals", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static int load_mode = -1;
        int lm = std::max(0, load_mode);
        ImGui::RadioButton("Heartbeat", &lm, 0);
        ImGui::RadioButton("Simple", &lm, 1);
        ImGui::RadioButton("Gaussian", &lm, 2);
        ImGui::RadioButton("Step", &lm, 3);
        ImGui::RadioButton("Smoothstep", &lm, 4);

        if (lm != load_mode)
        {
            percentage_filtered_ = 0.0f;
            load_mode = lm;
            int N = 500;
            switch (load_mode)
            {
                case 0:
                    ft1D_.load_signal(std::string(DATA_DIRECTORY) +
                                      std::string("signal-50Hz.dat"));
                    break;
                case 1:
                {
                    Signal s(N);
                    for (int i = 0; i < N; i++)
                    {
                        double freq = 10.0;
                        double x = i / static_cast<double>(N);
                        s[i] = 120 * cos(freq * 2.0 * M_PI * x) + 70;
                    }
                    ft1D_.set_signal(s);
                    break;
                }
                case 2:
                {
                    Signal s(N);
                    for (int i = 0; i < N; i++)
                    {
                        s[i] = 400 * exp(-(i - N / 2) * (i - N / 2) /
                                         static_cast<double>(N));
                    }
                    ft1D_.set_signal(s);
                    break;
                }
                case 3:
                {
                    Signal s(N);
                    for (int i = 0; i < N; i++)
                    {
                        s[i] = i < N / 2 ? -200 : 200;
                    }
                    ft1D_.set_signal(s);
                    break;
                }
                case 4:
                {
                    Signal s(N);
                    for (int i = 0; i < N; i++)
                    {
                        double t = std::min(
                            std::max((2 * (i - N / 2) + 100) / (200.0), 0.0),
                            1.0);
                        s[i] = 200 * ((t * t * (3.0 - 2.0 * t)) * 2.0 - 1.0);
                    }
                    ft1D_.set_signal(s);
                    break;
                }
                default:
                    break;
            }
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Filters", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static float k = 0;
        static float Fk = 10;
        static int addtoFk = 0;

        ImGui::Spacing();

        if (ImGui::Button("Reset"))
        {
            ft1D_.reset_filtered();
            percentage_filtered_ = 0.0f;
        }

        ImGui::SameLine();

        if (ImGui::Button("Rescale"))
        {
            rescale_ = true;
        }

        ImGui::Spacing();

        // calculate text height
        text_height_ = ImGui::GetContentRegionAvail().y;
        ImGui::Text("Choose k:");
        text_height_ -= ImGui::GetContentRegionAvail().y;

        ImGui::PushItemWidth(100);
        if (ImGui::SliderFloat("##kfreq", &k, 0,
                               ft1D_.get_signal_x().size() / 2, "%.0f",
                               ImGuiSliderFlags_Logarithmic))
        {
            addtoFk = 0;
        }
        ImGui::PopItemWidth();

        ImGui::Spacing();

        ImGui::Text("Add to F(k):");
        ImGui::PushItemWidth(100);
        if (ImGui::SliderInt("##kadd", &addtoFk, -100, 100, "%.0f"))
        {
            ft1D_.add_to_frequency_value(k, addtoFk);
            ft1D_.inverse_fourier_transform();
        }
        ImGui::PopItemWidth();

        ImGui::Spacing();

        ImGui::Text("Filter k:");
        if (ImGui::Button("Low-Pass"))
        {
            percentage_filtered_ = ft1D_.frequency_filter(k, true);
            ft1D_.inverse_fourier_transform();
        }

        if (ImGui::Button("High-Pass"))
        {
            percentage_filtered_ = ft1D_.frequency_filter(k, false);
            ft1D_.inverse_fourier_transform();
        }

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("Choose ||F(k)||");
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("##Fkfreq", &Fk, 0, 200, "%.1f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::PopItemWidth();

        ImGui::Spacing();

        ImGui::Text("Filter ||F(k)||:");
        if (ImGui::Button("Filter above"))
        {
            percentage_filtered_ = ft1D_.frequency_value_filter(Fk, true);
            ft1D_.inverse_fourier_transform();
        }

        if (ImGui::Button("Filter below"))
        {
            percentage_filtered_ = ft1D_.frequency_value_filter(Fk, false);
            ft1D_.inverse_fourier_transform();
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Draw Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Draw unfiltered", &draw_unfiltered_);
    }
}

//-----------------------------------------------------------------------------

void FourierViewer::imgui_2D_controls()
{
    if (ImGui::CollapsingHeader("Mode", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        static int hy = (int)hybrid_;
        if (ImGui::RadioButton("Filter Images", &hy, 0) ||
            ImGui::RadioButton("Hybrid Images", &hy, 1))
        {
            hybrid_ = (hy == 1);
            tex_filtered_signal_.texID = 0;
            tex_filtered_spectrum_.texID = 0;
            if (!hybrid_)
            {
                tex_spectrum_.texID = 0;
            }
            reload_ = true;
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (!hybrid_)
    {
        imgui_2D_controls_filter();
    }
    else
    {
        imgui_2D_controls_hybrid();
    }
}

//-----------------------------------------------------------------------------

void FourierViewer::imgui_2D_controls_filter()
{
    static const std::string dd = std::string(DATA_DIRECTORY);
    static std::string fn = dd + std::string("miezen_small.png");

    if (ImGui::CollapsingHeader("Images", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        static int load_mode = -1;
        int lm = std::max(0, load_mode);
        ImGui::RadioButton("Pixel-Pingu", &lm, 0);
        ImGui::RadioButton("Cats (small)", &lm, 1);
        ImGui::RadioButton("Cats (large)", &lm, 2);
        ImGui::RadioButton("Angry", &lm, 3);
        ImGui::RadioButton("Happy", &lm, 4);
        ImGui::RadioButton("Brick", &lm, 5);
        ImGui::RadioButton("Brick rotated", &lm, 6);

        if (lm != load_mode || reload_)
        {
            nearest_ = false;
            reload_ = false;
            load_mode = lm;

            switch (lm)
            {
                case 0:
                    fn = "pixel_pingu.png";
                    nearest_ = true;
                    break;
                case 1:
                    fn = "miezen_small.png";
                    break;
                case 2:
                    fn = "miezen.png";
                    break;
                case 3:
                    fn = "angry.png";
                    break;
                case 4:
                    fn = "happy.png";
                    break;
                case 5:
                    fn = "brick.png";
                    break;
                case 6:
                    fn = "brick_rotated.png";
                    break;
            }
            fn = dd + fn;

            if (ft2D_.load_signal(fn))
                create_texture_from_2D_signal(ft2D_.signal_, !colored_,
                                              tex_signal_);

            tex_spectrum_.texID = 0;
            tex_filtered_spectrum_.texID = 0;
            tex_filtered_signal_.texID = 0;
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("2D Filters", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        static int filter_type = -1;
        int ft = std::max(0, filter_type);
        ImGui::RadioButton("None", &ft, 0);
        ImGui::RadioButton("Gaussian ", &ft, 1);
        ImGui::RadioButton("Inv-Gaussian ", &ft, 2);
        if (ft != filter_type)
        {
            filter_type = ft;
            ft2D_.filter_type_ = static_cast<Fourier2D::FilterType>(ft);

            tex_filtered_signal_.texID = 0;
            tex_filtered_spectrum_.texID = 0;
        }

        if (ft >= 1)
        {
            static int kw = 10;
            ImGui::Text("Choose Kernel Width:");
            ImGui::PushItemWidth(100);
            if (ImGui::SliderInt("##kw", &kw, 1, 100, "%d",
                                 ImGuiSliderFlags_Logarithmic))
            {
                tex_filtered_signal_.texID = 0;
                tex_filtered_spectrum_.texID = 0;
            }
            ImGui::PopItemWidth();
            ft2D_.kernel_width_ = kw;
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        if (ImGui::Checkbox("Colored", &colored_))
        {
            ft2D_.set_colored(colored_);

            int nc = colored_ ? 3 : 1;
            second_signal_.set_num_components(nc);

            reload_ = true;
        }
    }

#ifndef __EMSCRIPTEN__
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Write to Image file",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        static int mm = 2;
        ImGui::RadioButton("Spectrum", &mm, 0);
        ImGui::RadioButton("Filtered Spectrum", &mm, 1);
        ImGui::RadioButton("Filtered Image", &mm, 2);

        ImGui::Spacing();

        if (ImGui::Button("Write"))
        {
            switch (mm)
            {
                case 0:
                    ft2D_.spectrum_.write("my_spec.png");
                    break;
                case 1:
                    ft2D_.filtered_spectrum_.write("my_filt_spec.png");
                    break;
                case 2:
                    ft2D_.filtered_signal_.write("my_filt_imag.png");
                    break;
            }
        }
    }
#endif
}

//-----------------------------------------------------------------------------

void FourierViewer::imgui_2D_controls_hybrid()
{
    static const std::string dd = std::string(DATA_DIRECTORY);

    if (ImGui::CollapsingHeader("Images##hybrid",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        static int hybrid_images = -1;
        int hi = std::max(0, hybrid_images);
        ImGui::RadioButton("Male-Female ", &hi, 0);
        ImGui::RadioButton("Female-Male ", &hi, 1);
        ImGui::RadioButton("Marylin-Kennedy", &hi, 4);
        ImGui::RadioButton("Angry-Happy", &hi, 2);
        ImGui::RadioButton("Happy-Angry", &hi, 3);
        if (hi != hybrid_images || reload_)
        {
            nearest_ = false;
            reload_ = false;
            hybrid_images = hi;

            std::string fn1, fn2;

            switch (hi)
            {
                case 2:
                    fn1 = "angry.png";
                    fn2 = "happy.png";
                    break;
                case 0:
                    fn1 = "m.png";
                    fn2 = "f.png";
                    break;
                case 4:
                    fn1 = "marylin.png";
                    fn2 = "jfk.png";
                    break;

                case 3:
                    fn2 = "angry.png";
                    fn1 = "happy.png";
                    break;
                case 1:
                    fn2 = "m.png";
                    fn1 = "f.png";
                    break;
            }

            fn1 = dd + fn1;
            fn2 = dd + fn2;

            if (ft2D_.load_signal(fn1))
                create_texture_from_2D_signal(ft2D_.signal_, !colored_,
                                              tex_signal_);

            if (second_signal_.load(fn2))
                create_texture_from_2D_signal(second_signal_, !colored_,
                                              tex_spectrum_);

            tex_filtered_spectrum_.texID = 0;
            tex_filtered_signal_.texID = 0;
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Settings##hybrid",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        ImGui::Text("Change Image Size:");
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("##kw", &image_size_, 0.1, 1.0, "%.2f");
        ImGui::PopItemWidth();

        // enable mouse wheel scrolling to change size
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel && tex_filtered_signal_.texID != 0)
        {
            image_size_ += wheel * 0.02;
            image_size_ = std::max(0.1f, std::min(1.0f, image_size_));
        }

        ImGui::Spacing();

        if (ImGui::Checkbox("Colored", &colored_))
        {
            ft2D_.set_colored(colored_);

            int nc = colored_ ? 3 : 1;
            second_signal_.set_num_components(nc);

            reload_ = true;
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Write to Image file",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        if (ImGui::Button("Write##hybrid"))
        {
            ft2D_.filtered_signal_.write("my_hybrid.png");
        }
    }
}

//-----------------------------------------------------------------------------

void FourierViewer::process_implot_1D()
{
    ImVec2 firstWindowSize = ImGui::GetWindowSize();
    ImVec2 firstWindowPos = ImGui::GetWindowPos();
    const int window_padding = ImGui::GetStyle().WindowPadding.y;

    ImVec2 secondWindowSize(
        width() / high_dpi_scaling() - firstWindowSize.x -
            firstWindowPos.x * 2 - window_padding,
        height() / high_dpi_scaling() - firstWindowPos.y * 2);
    ImVec2 secondWindowPos(
        firstWindowPos.x + firstWindowSize.x + window_padding,
        firstWindowPos.y);

    int plot_height = (secondWindowSize.y - 3 * window_padding) / 2;
    bool open = true;
    ImGui::SetNextWindowPos(secondWindowPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(secondWindowSize, ImGuiCond_Always);
    ImGui::Begin("1D Fourier", &open, ImGuiWindowFlags_NoTitleBar);

    if (!ft1D_.get_signal_x().empty())
    {
        ImPlotAxisFlags flag = ImPlotAxisFlags_None;
        if (rescale_)
        {
            flag = ImPlotAxisFlags_AutoFit;
            rescale_ = false;
        }

        if (ImPlot::BeginPlot("Signal", ImVec2(-1, plot_height), ImPlotFlags_Crosshairs))
        {
            ImPlot::SetupAxis(ImAxis_X1, "x", flag);
            ImPlot::SetupAxis(ImAxis_Y1, "Signal", flag);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -650, 550);

            const int N = ft1D_.get_signal_x().size();
            if (draw_unfiltered_)
            {
                ImPlot::SetNextLineStyle(ImVec4(0, 0, 0.9, 1), 2.0);
                ImPlot::PlotLine("unfiltered", ft1D_.get_signal_x().data(),
                                 ft1D_.get_signal_y().data(), N);
            }
            if (!ft1D_.get_signal_y(true).empty())
            {
                std::string label =
                    std::string("filtered ") +
                    std::to_string((int)(percentage_filtered_ * 100)) +
                    std::string("%");
                ImPlot::SetNextLineStyle(ImVec4(0.65, 0.1, 0, 1), 2);
                ImPlot::PlotLine(label.c_str(), ft1D_.get_signal_x().data(),
                                 ft1D_.get_signal_y(true).data(), N);
            }
            ImPlot::EndPlot();
        }
        ImGui::Spacing();
        if (ImPlot::BeginPlot("Spectrum", ImVec2(-1, plot_height), ImPlotFlags_Crosshairs))
        {
            ImPlot::SetupAxis(ImAxis_X1, "Frequency", flag);
            ImPlot::SetupAxis(ImAxis_Y1, "abs(Spectrum)", flag);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 150);

            std::string label =
                std::string("filtered ") +
                std::to_string((int)(percentage_filtered_ * 100)) +
                std::string("%##2");

            const int N = ft1D_.get_signal_x().size();
            ImPlot::SetNextFillStyle(ImVec4(0.65, 0.1, 0, 1), -1);
            ImPlot::PlotBars(label.c_str(), ft1D_.get_spectrum_x().data(),
                             ft1D_.get_spectrum_y().data(), N, 0.8);

            ImPlot::EndPlot();
        }
    }

    ImGui::End();
}

//-----------------------------------------------------------------------------

void FourierViewer::process_implot_2D()
{
    ImVec2 firstWindowSize = ImGui::GetWindowSize();
    ImVec2 firstWindowPos = ImGui::GetWindowPos();
    const int window_padding = ImGui::GetStyle().WindowPadding.y;

    ImVec2 secondWindowSize(
        width() / high_dpi_scaling() - firstWindowSize.x -
            firstWindowPos.x * 2 - window_padding,
        height() / high_dpi_scaling() - firstWindowPos.y * 2);
    ImVec2 secondWindowPos(
        firstWindowPos.x + firstWindowSize.x + window_padding,
        firstWindowPos.y);

    bool open = true;
    ImGui::SetNextWindowPos(secondWindowPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(secondWindowSize, ImGuiCond_Always);

    if (!hybrid_)
    {
        ImGui::Begin("2D Fourier Images", &open, ImGuiWindowFlags_NoTitleBar);
        process_implot_2D_filter();
        ImGui::End();
    }
    else
    {
        ImGui::Begin("2D Hybrid Images", &open, ImGuiWindowFlags_NoTitleBar);
        process_implot_2D_hybrid();
        ImGui::End();
    }
}

//-----------------------------------------------------------------------------

void FourierViewer::process_implot_2D_filter()
{
    const int window_padding = ImGui::GetStyle().WindowPadding.y;
    const int item_spacing = ImGui::GetStyle().ItemInnerSpacing.y;
    ImVec2 maxImageSize(
        ImGui::GetContentRegionAvail().x * 0.5f - window_padding * 4.0f - 2.0f,
        ImGui::GetContentRegionAvail().y * 0.5f - text_height_ -
            window_padding - 0.5f * item_spacing);

    const static ImageTexture &tex1 = tex_signal_;
    const static ImageTexture &tex2 = tex_spectrum_;
    const static ImageTexture &tex3 = tex_filtered_spectrum_;
    const static ImageTexture &tex4 = tex_filtered_signal_;

    ImVec2 ChildWindowSize(ImGui::GetContentRegionAvail().x * 0.5f,
                           ImGui::GetContentRegionAvail().y);
    float factor =
        std::min(maxImageSize.x / tex1.width, maxImageSize.y / tex1.height);
    ImVec2 imageSize = ImVec2(factor * tex1.width, factor * tex1.height);
    ImVec2 padSize(
        std::max(1.0f,
                 (ChildWindowSize.x - imageSize.x - 4 * window_padding) / 1.5f),
        ChildWindowSize.y);
    ChildWindowSize.x -= 1.5 * padSize.x + 2 * window_padding;

    ImGui::BeginChild("ChildPad1", padSize, false, ImGuiWindowFlags_None);
    ImGui::EndChild();
    ImGui::SameLine();

    float text_height = 0;
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
    ImGui::BeginChild("ChildL", ChildWindowSize, false, window_flags);
    {
        // find out text height
        text_height = ImGui::GetContentRegionAvail().y;
        ImGui::Text("Original Image");
        text_height -= ImGui::GetContentRegionAvail().y;

        // recalculate image height
        maxImageSize.y = maxImageSize.y - text_height - window_padding;

        ImGui::Image((void *)(intptr_t)tex1.texID, imageSize);

        ImGui::Text("Spectrum");
        if (tex2.texID != 0)
            ImGui::Image((void *)(intptr_t)tex2.texID, imageSize);
        else
        {
            if (tex1.texID != 0)
            {
                bool selected = true;
                ImVec2 alignment = ImVec2(0.5, 0.5);
                ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,
                                    alignment);
                ImGui::SetWindowFontScale(2.0);
                if (ImGui::Selectable("Transform", &selected,
                                      ImGuiSelectableFlags_None, imageSize))
                {
                    ft2D_.FT();
                    create_texture_from_2D_spectrum(ft2D_.spectrum_, !colored_,
                                                    tex_spectrum_);
                }
                ImGui::SetWindowFontScale(1.0);
                ImGui::PopStyleVar();
            }
            else
            {
                ImGui::BeginChild("Child2", imageSize);
                ImGui::EndChild();
            }
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginChild("ChildPad2", padSize, false, ImGuiWindowFlags_None);
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginChild("ChildR", ChildWindowSize, false, window_flags);
    {
        ImGui::Text("Filtered Image");
        if (tex4.texID != 0)
        {
            ImGui::Image((void *)(intptr_t)tex4.texID, imageSize);
        }
        else
        {
            if (tex3.texID != 0)
            {
                bool selected = true;
                ImVec2 alignment = ImVec2(0.5, 0.5);
                ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,
                                    alignment);
                ImGui::SetWindowFontScale(2.0);
                if (ImGui::Selectable("Inverse Transform", &selected,
                                      ImGuiSelectableFlags_None, imageSize))
                {
                    ft2D_.iFT();
                    create_texture_from_2D_signal(ft2D_.filtered_signal_,
                                                  !colored_,
                                                  tex_filtered_signal_);
                }
                ImGui::SetWindowFontScale(1.0);
                ImGui::PopStyleVar();
            }
            else
            {
                ImGui::BeginChild("Child4", imageSize);
                ImGui::EndChild();
            }
        }

        ImGui::Text("Filtered Spectrum");
        if (tex3.texID != 0)
            ImGui::Image((void *)(intptr_t)tex3.texID, imageSize);
        else
        {
            if (tex2.texID != 0)
            {
                bool selected = true;
                ImVec2 alignment = ImVec2(0.5, 0.5);
                ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,
                                    alignment);
                ImGui::SetWindowFontScale(2.0);
                if (ImGui::Selectable("Filter", &selected,
                                      ImGuiSelectableFlags_None, imageSize))
                {
                    ft2D_.filter();
                    create_texture_from_2D_spectrum(ft2D_.filtered_spectrum_,
                                                    !colored_,
                                                    tex_filtered_spectrum_);
                }
                ImGui::SetWindowFontScale(1.0);
                ImGui::PopStyleVar();
            }
            else
            {
                ImGui::BeginChild("Child3", imageSize);
                ImGui::EndChild();
            }
        }
    }
    ImGui::EndChild();
}

//-----------------------------------------------------------------------------

void FourierViewer::process_implot_2D_hybrid()
{
    ImVec2 sourceImageSize(ImGui::GetContentRegionAvail().x * 0.25f - 1.5 * 10,
                           ImGui::GetContentRegionAvail().y * 0.25f - 1.0 * 10);

    ImVec2 destImageSize(ImGui::GetContentRegionAvail().x * 0.75f - 1.5 * 10,
                         ImGui::GetContentRegionAvail().y - 2.0 * 10);

    const static ImageTexture &tex1 = tex_signal_;
    const static ImageTexture &tex2 = tex_spectrum_;
    const static ImageTexture &tex4 = tex_filtered_signal_;

    ImVec2 ChildLWindowSize(ImGui::GetContentRegionAvail().x * 0.25f,
                            ImGui::GetContentRegionAvail().y);

    ImVec2 ChildRWindowSize(ImGui::GetContentRegionAvail().x * 0.75f,
                            ImGui::GetContentRegionAvail().y);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
    ImGui::BeginChild("ChildL", ChildLWindowSize, false, window_flags);
    {
        float factor = std::min(sourceImageSize.x / tex1.width,
                                sourceImageSize.y / tex1.height);
        ImVec2 imageSize = ImVec2(factor * tex1.width, factor * tex1.height);

        ImGui::Text("Image1 (low freq.)");
        ImGui::Image((void *)(intptr_t)tex1.texID, imageSize);

        ImGui::Text("Image2 (high freq.)");
        ImGui::Image((void *)(intptr_t)tex2.texID, imageSize);
    }
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginChild("ChildR", ChildRWindowSize, false, window_flags);
    {
        float factor = std::min(destImageSize.x / tex1.width,
                                destImageSize.y / tex1.height);
        ImVec2 imageSize(factor * tex1.width, factor * tex1.height);
        float offset = 1.0 / image_size_ - 1.0;
        ImVec2 leftTop(-offset, -offset);
        ImVec2 bottomRight(1.0 + offset, 1.0 + offset);

        ImGui::Text("Hybrid Image");
        if (tex4.texID != 0)
            ImGui::Image((void *)(intptr_t)tex4.texID, imageSize, leftTop,
                         bottomRight);
        else
        {
            bool selected = true;
            ImVec2 alignment = ImVec2(0.5, 0.5);
            ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, alignment);
            ImGui::SetWindowFontScale(2.0);
            if (ImGui::Selectable("Create Hybrid", &selected,
                                  ImGuiSelectableFlags_None, imageSize))
            {
                ft2D_.hybrid(second_signal_);

                create_texture_from_2D_signal(ft2D_.filtered_signal_, !colored_,
                                              tex_filtered_signal_);
            }
            ImGui::SetWindowFontScale(1.0);
            ImGui::PopStyleVar();
        }
    }
    ImGui::EndChild();
}

//-----------------------------------------------------------------------------

void FourierViewer::create_texture_from_2D_signal(
    const Signal2D &signal, const bool greyscale,
    FourierViewer::ImageTexture &out_tex)
{
    auto &tex = out_tex;
    // delete old texture
    glDeleteTextures(1, &tex.texID);

    // setup new texture
    glGenTextures(1, &tex.texID);
    glBindTexture(GL_TEXTURE_2D, tex.texID);

    GLint loadFormat = GL_RGB;
    size_t w = signal.width_;
    size_t h = signal.height_;

    unsigned char (*convert)(double) = [](double in) {
        return static_cast<unsigned char>(std::min(255.0, 255.0 * in));
    };

    // convert greyscale to rgb
    unsigned char *data = new unsigned char[w * h * 3];
    if (greyscale)
    {
        for (int i = 0; i < w * h; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                data[3 * i + c] = convert(signal.data()[i]);
            }
        }
    }
    else
    {
        for (int i = 0; i < w * h; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                data[3 * i + c] = convert(signal.data()[i + c * w * h]);
            }
        }
    }

    // upload texture data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, loadFormat,
                 GL_UNSIGNED_BYTE, data);

    // set texture parameters
    if (nearest_)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
#ifndef __EMSCRIPTEN__
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
#else
    // unfortunately, clamp to border doesn't work in WebGL
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#endif

    tex.width = signal.width_;
    tex.height = signal.height_;

    delete[] data;
}

//-----------------------------------------------------------------------------

void FourierViewer::create_texture_from_2D_spectrum(
    const Spectrum2D &spectrum, const bool greyscale,
    FourierViewer::ImageTexture &out_tex)
{
    auto &tex = out_tex;
    // delete old texture
    glDeleteTextures(1, &tex.texID);

    // setup new texture
    glGenTextures(1, &tex.texID);
    glBindTexture(GL_TEXTURE_2D, tex.texID);

    GLint loadFormat = GL_RGB;
    size_t w = spectrum.width_;
    size_t h = spectrum.height_;

    unsigned char (*convert)(complex) = [](complex in) {
        return static_cast<unsigned char>(
            std::min(255.0, 255.0 * std::abs(in)));
    };

    // convert greyscale to rgb
    unsigned char *data = new unsigned char[w * h * 3];
    if (greyscale)
    {
        for (int i = 0; i < w * h; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                data[3 * i + c] = convert(spectrum.data()[i]);
            }
        }
    }
    else
    {
        for (int i = 0; i < w * h; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                data[3 * i + c] = convert(spectrum.data()[i + c * w * h]);
            }
        }
    }

    // upload texture data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, loadFormat,
                 GL_UNSIGNED_BYTE, data);

    // set texture parameters
    if (nearest_)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    tex.width = spectrum.width_;
    tex.height = spectrum.height_;

    delete[] data;
}

//=============================================================================
