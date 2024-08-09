cmake_minimum_required(VERSION 3.6)
project(ED_Lib)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT} -fPIC")

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

option(BUILD_BINDING "build python binding" ON)
if(BUILD_BINDING)
  add_subdirectory(pybind11)
  pybind11_add_module(_edlib binding.cpp)
  target_link_libraries(_edlib PRIVATE EDLib ${OpenCV_LIBS})
  install(TARGETS _edlib DESTINATION .)
endif(BUILD_BINDING)
