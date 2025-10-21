cmake_minimum_required(VERSION 3.15)
project(vasapy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/parallel-hashmap"
)

pybind11_add_module(_vasapy
    src/dict.cpp
    src/set.cpp
    src/vasapy.cpp
)

set_target_properties(_vasapy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vasapy)

target_include_directories(_vasapy PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/parallel-hashmap"
)

target_compile_options(_vasapy PRIVATE -O3 -march=native -ffast-math)
install(TARGETS _vasapy LIBRARY DESTINATION vasapy)
