cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(cutlass_run LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Set CUTLASS directory
if(NOT DEFINED CUTLASS_DIR)
    set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "CUTLASS Repository Directory")
endif()

# Include CUTLASS
include("${CUTLASS_DIR}/CUDA.cmake")

# Create the executable first
file(GLOB CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)
message( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
message( STATUS "CUDA_SOURCES: ${CUDA_SOURCES}" )
add_executable(cutlass_run ${CUDA_SOURCES})

# Apply CUTLASS compilation options
# cutlass_apply_standard_compile_options(cutlass_run)

# Add CUTLASS include directories
target_include_directories(cutlass_run PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUTLASS_DIR}/examples/common
)

# Set CUDA properties
set_target_properties(cutlass_run PROPERTIES
    CUDA_ARCHITECTURES "89;"
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(cutlass_run
    CUDA::cudart
    CUDA::cublas
)