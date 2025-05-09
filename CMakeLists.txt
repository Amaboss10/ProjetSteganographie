cmake_minimum_required(VERSION 3.5)
project(ProjetSteganographieBase LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

# Résout le warning CMP0104 et spécifie les architectures CUDA cibles
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 86)

# Exécutable avec tous les fichiers source nécessaires
add_executable(ProjetSteganographieBase
    main.cpp
    utils/image.cpp
    utils/raw_image_loader.cpp
    utils/decode_cpu.cpp
    utils/encode_cpu.cpp
    utils/encode_naive.cu
    utils/find_sentinels_naive.cu

    utils/image.hpp
    utils/raw_image_loader.hpp
    utils/decode_cpu.hpp
    utils/encode_cpu.hpp
    utils/encode_naive.hpp
    utils/find_sentinels_naive.hpp
    utils/stb/stb_image.h
    utils/stb/stb_image_write.h
)

# Behind this comment, the CMake instructions to copy the images folder
# to the build folder when the project is built.
set(input_folder "${CMAKE_CURRENT_SOURCE_DIR}/images")
set(output_folder "${CMAKE_BINARY_DIR}/images")

file(GLOB_RECURSE folder_content ${input_folder}/*.*)
add_custom_target(  Images_ProjetSteganographieBase COMMAND ${CMAKE_COMMAND} -E copy_directory ${input_folder}/ ${output_folder}/)
set_property(TARGET Images_ProjetSteganographieBase APPEND PROPERTY SOURCES ${folder_content})
add_dependencies(ProjetSteganographieBase Images_ProjetSteganographieBase)