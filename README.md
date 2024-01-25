Fouriertransform
================

This package contains exercise code for the lecture "Computer Graphics" by Prof. Dr. Mario Botsch, TU Dortmund.


Documentation
-------------

A pre-built HTML documentation can be found in `doc/index.html` and can be opened via any web browser.


Prerequisites
-------------

We use [CMake](www.cmake.org) for setting up build environments. You can download it [here](https://cmake.org/download/) or by using your favourite package manager.
You will also need a C++ compiler. Depending on your OS, the typical choices are:

  * Linux - GCC/G++, usually pre-installed (terminal command: `sudo apt install build-essential`)
  * MacOS - Clang, comes with the IDE "XCode" (terminal command: `xcode-select --install`)
  * Windows - MSVC, can be installed alongside the IDE "Visual Studio Community" (see below) or separately [here](https://visualstudio.microsoft.com/downloads/#other) under "All Downloads" --> "Tools for Visual Studio" --> "Build Tools for Visual Studio 2019"

We highly recommand to use some kind of IDE. Prominent examples are:

 - [XCode](https://developer.apple.com/xcode/) (MacOS)
 - [Visual Studio Community](https://visualstudio.microsoft.com/de/vs/community/) (Windows)
 - [Visual Studio Code](https://code.visualstudio.com/) (Linux, Windows, MacOS)
 - [Jetbrains CLion](https://www.jetbrains.com/de-de/clion/) (Linux, Windows, MacOS)

Below, we provide examples for setting up, compiling and running the project via command line, XCode, VSCommunity and VSCode.


Building on Linux via Command Line (no IDE)
-------------------------------------------

Execute the following commands in the exercise's top-level directory:

    mkdir build
    cd build
    cmake ..
    make

The last command (`make`) compiles the application. Re-run it whenever you have added/changed code in order to re-compile.

For running the code via command line use

    ./fourier


Building on MacOS (XCode)
--------------------------

Execute the following commands in the exercise's top-level directory:

    mkdir xcode
    cd xcode
    cmake -G Xcode ..

Open the generated .xcodeproj file. Inside XCode, select the fourier-executable in the top bar next to the stop button. You can run the code by pressing the play button.

Building on Windows (Visual Studio Community)
---------------------------------------------

* In order to get the MSVC C++ compiler, make sure that you check "Desktop development with C++" during installation of [VSCommunity](https://visualstudio.microsoft.com/de/vs/community/)
* Create an empty build folder inside the project's top-level directory
* Start cmake-gui.exe (located in cmake's bin folder)
* Specify the top-level directory as source directory (button Browse source...)
* Specify the previously created build folder as build directory (button Browse build...)
* Select "Configure" using your Visual Studio Version as option.
* When configuration is finished, select "Generate".
* Start Visual Studio Community
* Open the project via File -> open -> project -> .sln in build folder
* In the project explorer window on the right, right-click the project (fourier) and set it as startup-project
* Switch to release mode
* Hit CTRL + F5 to build and run (or CTRL + SHIFT + B to build)


Building via VSCode
-------------------

There are a lot of useful extensions for VSCode to shape it the way you like.
The required extensions for C++ development and Cmake support are "C/C++" and "Cmake Tools". Extensions can be found in the extensions tab on the left.
Once this is done, you can set up the project:

 * Start VSCode
 * Open the project via File --> Open Folder and select the exercise's top-level directory containing the `CMakeLists.txt` file and accept by clicking `OK`
 * In the bottom bar, click on `CMake`, choose your compiler and select `Release` mode
 * In the bottom bar, click on  `[all]` and select the right target (fourier)
 * Still in the bottom bar, there are buttons for building (Build) and running (play symbol)

Have a look at the VSCode [documentation](https://code.visualstudio.com/docs/cpp/introvideos-cpp) for further details.


Navigation
----------

Use our GUI to interact with the viewer.


Recommended order for the exercises
-----------------------------------

* **One-Dimensional Fourier Transformations:**
  Start by getting the the one-dimensional Fourier transformation to work. In `src/fourier1D.cpp`, finish the methods `fourier_transform` and `inverse_fourier_transform`. 
  Note the use of complex numbers, which are part of the C++ standard library.
  Implement some filtering methods by completing the methods `frequency_filter` and `frequency_value_filter`.

* **Two-Dimensional Fourier Transformations:**
  Once you are comfortable with one-dimensional Fourier transformations, move on to the two-dimensional case. 
  Implement the methods `fourier_transform` and `inverse_fourier_transform` in `src/fourier2D.cpp`. 
  In order to avoid complexity `O(N^4)`, separate the stacked sums where possible. Use OpenMP for parallelization.
  Implement a 2D gaussian filter by completing the method `filter_gaussian`.

* **Hybrid Images:**
  Lastly, try to create hybrid images. This should already work, if you finished the last part. The implementation can be found in the method `hybrid` in `src/fourier2D.cpp`.

