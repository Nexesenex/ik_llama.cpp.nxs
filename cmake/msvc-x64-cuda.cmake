# MSVC toolchain for llama.cpp with CUDA
# Usage: cmake -G "Visual Studio 18 2026" -DCMAKE_TOOLCHAIN_FILE=cmake/msvc-x64-cuda.cmake -B build -S .
# Note: Use VS generator (not Ninja) for CUDA compatibility.
# Requires: MSVS 18 and CUDA 12.9

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# MSVS 2026
set(VS_ROOT "P:/Program Files/Microsoft Visual Studio/18/Community")
set(MSVC_VERSION "14.51.36014")

# CUDA installation
set(CUDA_ROOT "P:/NVIDIAGPUCT/CUDA/v12.9" CACHE PATH "CUDA installation directory")

# Windows SDK
set(WINSDK_ROOT "P:/Windows Kits/10")
set(WINSDK_VERSION "10.0.26100.0")

# Use MSVC compiler
set(CMAKE_C_COMPILER   "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/cl.exe")

# Skip CMake compiler checks
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Force compiler ID and version
set(CMAKE_C_COMPILER_ID "MSVC")
set(CMAKE_CXX_COMPILER_ID "MSVC")
set(CMAKE_C_COMPILER_VERSION "${MSVC_VERSION}")
set(CMAKE_CXX_COMPILER_VERSION "${MSVC_VERSION}")
set(CMAKE_VC_COMPILER_VERSION "${MSVC_VERSION}")

# Skip CMake compiler checks
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Use MSVC link.exe
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

# CUDA settings - must be set before project()
set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc.exe")
set(CMAKE_CUDA_HOST_COMPILER "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/cl.exe")

# Also set these for CMake to find CUDA
set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_ROOT}")
set(CUDA_TOOLKIT_ROOT "${CUDA_ROOT}")

# MSVC bin directory (needed by nvcc to find cl.exe)
set(MSVC_BIN "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64")

# Include and lib paths
set(MSVC_INCLUDE    "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/include")
set(MSVC_LIB        "${VS_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/lib/x64")
set(WINSDK_INC      "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/ucrt;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/shared;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/um")
set(WINSDK_LIB      "${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/ucrt/x64;${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/um/x64")
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
    "${CUDA_LIB}"
)

# Prepend MSVC bin to PATH so nvcc can find cl.exe
set(ENV{PATH} "${MSVC_BIN};$ENV{PATH}")

# =============================================================================
# C/C++ specific flags
# =============================================================================

# Base flags for MSVC
set(CMAKE_C_FLAGS_INIT "/O2 /GL /Gy /MP /EHsc /GS /fp:precise /std:c11")
set(CMAKE_CXX_FLAGS_INIT "/O2 /GL /Gy /MP /EHsc /GS /fp:precise /std:c++20")

# Add CPU-specific flags
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} /arch:AVX2")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} /arch:AVX2")

# =============================================================================
# CUDA specific flags
# =============================================================================

# CUDA compile flags
set(CMAKE_CUDA_FLAGS_INIT "-D_WINDOWS -Xcompiler=\" /GR /EHsc\"")

# =============================================================================
# OpenMP configuration
# =============================================================================

# Use MSVS OpenMP
set(OpenMP_C_FLAGS "-openmp")
set(OpenMP_CXX_FLAGS "-openmp")

message(STATUS "MSVC + CUDA toolchain loaded")
message(STATUS "  MSVC root: ${VS_ROOT}")
message(STATUS "  CUDA root: ${CUDA_ROOT}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Linker: ${CMAKE_LINKER}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  OpenMP: MSVC OpenMP")
message(STATUS "  Windows SDK: ${WINSDK_ROOT} (${WINSDK_VERSION})")
