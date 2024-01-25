file(GLOB_RECURSE SRCS ./*.cpp)
file(GLOB_RECURSE HDRS ./*.h)

add_executable(fourier ${SRCS} ${HDRS})
target_link_libraries(fourier pmp implot)

if (EMSCRIPTEN)
    set_target_properties(fourier PROPERTIES LINK_FLAGS "--shell-file ${PROJECT_SOURCE_DIR}/external/pmp/shell.html --preload-file ${PROJECT_SOURCE_DIR}/data/@data/")
endif()
