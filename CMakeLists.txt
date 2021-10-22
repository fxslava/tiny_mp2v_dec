cmake_minimum_required(VERSION 3.16)
project(product_tiny_mp2v_enc)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tiny_decoder)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
