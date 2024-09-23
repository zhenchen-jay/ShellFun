cmake_minimum_required(VERSION 3.1)
project(ShellFun)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(MSVC_RUNTIME "dynamic")
set(LIBIGL_EIGEN_VERSION 3.3.7 CACHE STRING "Eigen version")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(libigl)
include(libshell)
include(polyscope)
include(spdlog)
include(CLI11)

igl_include(restricted triangle)

if(MSVC)
	add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()

# Set up include directories for all examples
include_directories(${libshell_SOURCE_DIR}/include)
include_directories(${libshell_SOURCE_DIR}/optimization/include)

# Add source files from optimization folder to the build
file(GLOB OPTIMIZATION_SRC_FILES ${libshell_SOURCE_DIR}/optimization/src/*.cpp)

add_subdirectory(examples)

