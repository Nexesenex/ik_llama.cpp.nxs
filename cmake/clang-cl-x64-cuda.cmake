# Toolchain file for standalone LLVM/clang-cl x64 builds with CUDA.
# Inherits all C/CXX settings from clang-cl-x64.cmake

include("${CMAKE_CURRENT_LIST_DIR}/clang-cl-x64.cmake")

set(CMAKE_CUDA_COMPILER "P:/NVIDIAGPUCT/CUDA/v12.9/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")
set(CUDAToolkit_BIN_DIR "P:/NVIDIAGPUCT/CUDA/v12.9/bin")
set(CMAKE_CUDA_HOST_COMPILER "C:/LLVM/bin/clang-cl.exe")

# Pass -mavxvnni through nvcc to the host compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-mavxvnni'")