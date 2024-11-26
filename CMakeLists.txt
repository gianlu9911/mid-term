cmake_minimum_required(VERSION 3.5.0)
project(kmeans VERSION 0.1.0 LANGUAGES C CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(include)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add executable targets
add_executable(experiment main.cpp)
add_executable(experimentAoS main2.cpp)

# Enable optimization flags for the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize -fopenmp")

# Add custom compile definitions and OpenMP
target_compile_definitions(experiment PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(experimentAoS PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

# Link OpenMP and set compiler flags
target_link_libraries(experiment PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(experimentAoS PRIVATE OpenMP::OpenMP_CXX)
