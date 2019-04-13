cmake_minimum_required(VERSION 2.8.12)
project(cohomopt)

add_subdirectory(pybind11)
pybind11_add_module(cohomopt cohomopt.cpp)

