cmake_minimum_required(VERSION 3.13)
project(KernelConvSeq)

set(CMAKE_CXX_STANDARD 14)

add_executable(KernelConvSeq main.cpp PPM.cpp PPM.h Image.cpp Image.h Utils.h)