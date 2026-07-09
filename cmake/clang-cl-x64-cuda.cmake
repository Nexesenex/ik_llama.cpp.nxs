# Toolchain file for standalone LLVM/clang-cl x64 builds with CUDA.
# Inherits all C/CXX settings from clang-cl-x64.cmake

include("${CMAKE_CURRENT_LIST_DIR}/clang-cl-x64.cmake")

# CUDA: use the CUDA 12.9 nvcc as the CUDA compiler.
# clang-cl handles the host-side, nvcc handles the device-side.
set(CMAKE_CUDA_COMPILER "P:/NVIDIAGPUCT/CUDA/v12.9/bin/nvcc.exe")

# CUDA architectures for the RTX 3090 + A4000 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

# CUDA host compiler must match our clang-cl
set(CMAKE_CUDA_HOST_COMPILER "C:/LLVM/bin/clang-cl.exe")

set(CUDAToolkit_BIN_DIR "P:/NVIDIAGPUCT/CUDA/v12.9/bin")

# NVCC host compiler flags — pass -mavxvnni so host-code compiled by nvcc
# (which will pass it through to clang-cl) also gets VNNI support.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-mavxvnni'")