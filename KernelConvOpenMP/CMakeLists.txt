cmake_minimum_required(VERSION 3.13)
project(KernelConvOpenMP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(KernelConvOpenMP main.cpp Image.cpp Image.h PPM.cpp PPM.h Utils.h)