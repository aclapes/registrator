cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(registrator)

find_package(OpenCV REQUIRED)

add_executable (registrator registrator.cpp)
target_link_libraries (registrator ${OpenCV_LIBS})
