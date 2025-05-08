cmake_minimum_required(VERSION 3.5)
project(ProjetSteganographieBase)

set(CMAKE_CXX_STANDARD 17)

add_executable(ProjetSteganographieBase main.cpp
        utils/image.hpp
        utils/image.cpp
        utils/raw_image_loader.cpp
        utils/raw_image_loader.hpp
        utils/stb/stb_image.h
        utils/stb/stb_image_write.h
        utils/encode_cpu.cpp
        utils/encode_cpu.hpp
        utils/decode_cpu.cpp
        utils/decode_cpu.hpp
)

# Behind this comment, the CMake instructions to copy the images folder
# to the build folder when the project is built.
set(input_folder "${CMAKE_CURRENT_SOURCE_DIR}/images")
set(output_folder "${CMAKE_BINARY_DIR}/images")

file(GLOB_RECURSE folder_content ${input_folder}/*.*)
add_custom_target(  Images_ProjetSteganographieBase COMMAND ${CMAKE_COMMAND} -E copy_directory ${input_folder}/ ${output_folder}/)
set_property(TARGET Images_ProjetSteganographieBase APPEND PROPERTY SOURCES ${folder_content})
add_dependencies(ProjetSteganographieBase Images_ProjetSteganographieBase)