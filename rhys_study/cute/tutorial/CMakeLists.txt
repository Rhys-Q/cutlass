
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0 -g -lineinfo") 
# set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
# Find CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(CUTLASS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
include(${CUTLASS_SOURCE_DIR}/CUDA.cmake)
include_directories(${CUTLASS_SOURCE_DIR}/include)
include_directories(${CUTLASS_SOURCE_DIR}/tools/util/include)
message("CUTLASS_SOURCE_DIR: ${CUTLASS_SOURCE_DIR}")

# add_subdirectory(hopper)
# add_subdirectory(blackwell)



# Create the executable first
file(GLOB CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/layout.cu"
)
message( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
message( STATUS "CUDA_SOURCES: ${CUDA_SOURCES}" )

add_executable(cutlass_run ${CUDA_SOURCES})


# Add CUTLASS include directories
message( STATUS "CUTLASS_DIR: ${CUTLASS_SOURCE_DIR}" )
target_include_directories(cutlass_run PRIVATE
    ${CUTLASS_SOURCE_DIR}/include
    ${CUTLASS_SOURCE_DIR}/tools/util/include
    ${CUTLASS_SOURCE_DIR}/examples/common
)

# Set CUDA properties
set_target_properties(cutlass_run PROPERTIES
    CUDA_ARCHITECTURES "89;80"
    # CUDA_SEPARABLE_COMPILATION ON
)

# # Link libraries
# target_link_libraries(cutlass_run
#     CUDA::cudart
#     CUDA::cublas
# )