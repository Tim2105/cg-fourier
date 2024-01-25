##############################################################################

cmake_minimum_required(VERSION 3.12)
project(cg_fourier)

set(CMAKE_PROJECT_VERSION 1.0.0)
set(CMAKE_PROJECT_VERSION_MAJOR 1)
set(CMAKE_PROJECT_VERSION_MINOR 0)
set(CMAKE_PROJECT_VERSION_PATCH 0)


##############################################################################
# paths
##############################################################################

set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})


##############################################################################
# dependencies
##############################################################################

# dependencies
cmake_policy(SET CMP0072 NEW)
find_package(OpenGL REQUIRED)
find_package(OpenMP)

##############################################################################
# compiler flags
##############################################################################

# we need C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#our data
if (EMSCRIPTEN)
    add_definitions(-DDATA_DIRECTORY="./data/")
else()
    add_definitions(-DDATA_DIRECTORY="${CMAKE_SOURCE_DIR}/data/")
endif()


if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
  add_compile_options(/wd4267) # conversion from size_t
  add_compile_options(/wd4244) # conversion from double to float
  add_compile_options(/wd4305) # truncation from double to float
  add_compile_options(/wd26451) # conversion float to double
  add_compile_options(/wd4005) # macro redefinition
endif()

# multi-core parallelization through OpenMP
if (OPENMP_FOUND AND NOT APPLE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (EMSCRIPTEN)
    add_compile_options(--no-heap-copy)
    add_link_options("SHELL:-sWASM=1 -sUSE_WEBGL2=1 -sUSE_GLFW=3 -sALLOW_MEMORY_GROWTH=1 -sALLOW_TABLE_GROWTH=1 -sSTACK_SIZE=5MB -sDISABLE_EXCEPTION_CATCHING=0")
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
endif()


##############################################################################
# build type
##############################################################################

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
endif()


##############################################################################
# STB 
# (place *before* GLFW since GLFW has an old copy of stb_image_write.h)
##############################################################################

set(STB_SOURCE_DIR "external/stb_image")
include_directories(${STB_SOURCE_DIR})
add_subdirectory(${STB_SOURCE_DIR})


##############################################################################
# GLFW
##############################################################################

if(NOT EMSCRIPTEN)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
  set(GLFW_INSTALL OFF CACHE BOOL "")
  add_subdirectory(external/glfw)
  include_directories(external/glfw/include
                      external/glfw/deps)
endif()


##############################################################################
# GLEW
##############################################################################

if(NOT EMSCRIPTEN)
  include_directories(external/glew/include)
  add_definitions(-DGLEW_STATIC)
  add_library(glew STATIC
              external/glew/src/glew.c
              external/glew/include)
  target_link_libraries(glew ${GLFW_LIBRARIES})
endif()


##############################################################################
# imgui
##############################################################################

include_directories(external/imgui)
add_subdirectory(external/imgui)

##############################################################################
# implot
##############################################################################

include_directories(external/implot)
add_subdirectory(external/implot)


##############################################################################
# pmp
##############################################################################

set(PMP_SOURCE_DIR "external/pmp")
include_directories(external)
add_subdirectory(external/pmp)


##############################################################################
# source code directories
##############################################################################


include(AddFileDependencies)
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(/usr/local/include/)
include_directories(${OPENGL_INCLUDE_DIRS})

add_subdirectory(src)


##############################################################################



