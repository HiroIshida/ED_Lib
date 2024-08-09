cmake_minimum_required(VERSION 3.6)
project(ED_Lib)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(EDLib STATIC
        ED.cpp
        EDCircles.cpp
        EDColor.cpp
        EDLines.cpp
        EDPF.cpp
        NFA.cpp
        NFA.h)

add_executable(alltest test.cpp)
target_link_libraries(alltest ${OpenCV_LIBS} EDLib)
