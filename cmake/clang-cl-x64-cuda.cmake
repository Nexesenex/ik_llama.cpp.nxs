# Clang-CL toolchain for llama.cpp with LLVM OpenMP 5.1 + CUDA
# Usage: cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/clang-cl-x64-cuda.cmake -B build -S .
# Requires: LLVM/Clang in VS 2026 and CUDA 12.9

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# MSVC 2026 + LLVM (Clang) installation
set(VS_ROOT "P:/Program Files/Microsoft Visual Studio/18/Community")
set(MSVC_VERSION "14.50.35717")
set(LLVM_ROOT "${VS_ROOT}/VC/Tools/Llvm")
set(LLVM_BIN "${LLVM_ROOT}/x64/bin")

# CUDA installation
set(CUDA_ROOT "P:/NVIDIAGPUCT/CUDA/v12.9" CACHE PATH "CUDA installation directory")

# Windows SDK
set(WINSDK_ROOT "P:/Windows Kits/10")
set(WINSDK_VERSION "10.0.26100.0")

# Use Clang-CL from VS 2026 for C/C++ compilation
set(CMAKE_C_COMPILER   "${LLVM_BIN}/clang-cl.exe")
set(CMAKE_CXX_COMPILER "${LLVM_BIN}/clang-cl.exe")

# Use MSVC link.exe (better Windows compatibility than lld-link)
set(CMAKE_LINKER "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/link.exe")

# Resource compiler from MSVC
set(CMAKE_RC_COMPILER "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/rc.exe")

# Manifest tool from Windows SDK
set(CMAKE_MT "${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64/mt.exe")

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture flags for RTX 3090 and RTX A4000 (sm_86 = Ampere)
set(CMAKE_CUDA_ARCHITECTURES "86")

# CUDA settings
set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc.exe")

# MSVC bin directory (needed by nvcc to find cl.exe)
set(MSVC_BIN "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64")

# Include and lib paths
set(MSVC_INCLUDE    "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/include")
set(MSVC_LIB        "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/lib/x64")
set(WINSDK_INC      "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/ucrt;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/shared;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/um")
set(WINSDK_LIB      "${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/ucrt/x64;${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/um/x64")
set(LLVM_LIB        "${LLVM_ROOT}/x64/lib")
set(CUDA_INC        "${CUDA_ROOT}/include")
set(CUDA_LIB        "${CUDA_ROOT}/lib/x64")

# Common include paths
include_directories(
    "${MSVC_INCLUDE}"
    ${WINSDK_INC}
    "${CUDA_INC}"
)

# Common library paths
link_directories(
    "${MSVC_LIB}"
    ${WINSDK_LIB}
    "${LLVM_LIB}"
    "${CUDA_LIB}"
)

# Prepend MSVC bin to PATH so nvcc can find cl.exe
set(ENV{PATH} "${MSVC_BIN};$ENV{PATH}")

# CUDA compile flags (matching existing working config)
set(CMAKE_CUDA_FLAGS "-D_WINDOWS -Xcompiler=\" /GR /EHsc\"")

message(STATUS "Clang-CL + CUDA toolchain loaded")
message(STATUS "  LLVM root: ${LLVM_ROOT}")
message(STATUS "  CUDA root: ${CUDA_ROOT}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Linker: ${CMAKE_LINKER}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  Windows SDK: ${WINSDK_ROOT} (${WINSDK_VERSION})")
