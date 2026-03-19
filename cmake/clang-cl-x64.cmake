# Clang-CL toolchain for llama.cpp with LLVM OpenMP 5.1
# Usage: cmake -G "Visual Studio 18 2026" -DCMAKE_TOOLCHAIN_FILE=cmake/clang-cl-x64.cmake -B build -S .
# Note: Use VS generator for best compatibility.
# Requires: LLVM 22+ installed at C:/Program Files/LLVM

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# LLVM/Clang installation
set(LIBCLANG_ROOT "C:/Program Files/LLVM" CACHE PATH "LLVM installation directory")

# MSVC 2026 installation (for libraries and Windows SDK)
set(MSVC_ROOT "P:/Program Files/Microsoft Visual Studio/18/Community")
set(MSVC_VERSION "14.50.35717")

# Windows SDK
set(WINSDK_ROOT "P:/Windows Kits/10")
set(WINSDK_VERSION "10.0.26100.0")

# Use Clang-CL as the compiler (compatible with MSVC project files)
set(CMAKE_C_COMPILER   "${LIBCLANG_ROOT}/bin/clang-cl.exe")
set(CMAKE_CXX_COMPILER "${LIBCLANG_ROOT}/bin/clang-cl.exe")

# Skip CMake compiler checks - we trust clang-cl works
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Linker: use MSVC link.exe for better Windows compatibility
set(CMAKE_LINKER "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/link.exe")

# Resource compiler from MSVC
set(CMAKE_RC_COMPILER "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/rc.exe")

# Manifest tool from Windows SDK
set(CMAKE_MT "${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64/mt.exe")

# Architecture and compiler identification
set(CMAKE_VC_COMPILER_ID Clang)
set(MSVC_CXX_COMPILER_ID Clang)
set(CMAKE_C_COMPILER_ID Clang)
set(CMAKE_CXX_COMPILER_ID Clang)

# Tell CMake this is a Clang compiler (not MSVC)
set(CMAKE_COMPILER_IS_GNUCXX OFF)
set(CMAKE_COMPILER_IS_GNUCC OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include and lib paths
set(MSVC_INCLUDE    "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/include")
set(MSVC_LIB        "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/lib/x64")
set(WINSDK_INC      "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/ucrt;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/shared;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/um")
set(WINSDK_LIB      "${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/ucrt/x64;${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/um/x64")
set(LLVM_LIB        "${LIBCLANG_ROOT}/lib")

# Common include paths - standard order: MSVC, Windows SDK, then LLVM (for OpenMP only)
include_directories(
    "${MSVC_INCLUDE}"
    ${WINSDK_INC}
)

# Common library paths
link_directories(
    "${MSVC_LIB}"
    ${WINSDK_LIB}
    "${LLVM_LIB}"
)

# Optimization flags for performance (Clang-CL compatible)
# NOTE: Do NOT use /Gm (minimal rebuild) with /std:c++20
add_compile_options(
    /O2
    /GL
    /Gy
    /MP
    /EHsc
    /GS
    /fp:precise
    /openmp:llvm
    /std:c++20
    /arch:AVX2
)

# Disable some Clang warnings that are overly noisy
add_compile_options(
    /W4
    /WX-
)

# Add required flags for ggml.c and SIMD support
add_compile_options(
    -fno-finite-math-only
    -mavx2
    -mbmi2
    -mfma
    -mavxvnni
    -mavxifma
    -mcmpccxadd
)

# Disable the __FINITE_MATH_ONLY__ check in ggml.c
add_definitions(-D__FINITE_MATH_ONLY__=0)

# OpenMP settings - use LLVM's libomp (OpenMP 5.1)
# IMPORTANT: Clang-CL uses /openmp:llvm (Windows style)
set(OpenMP_CXX_FLAGS "/openmp:llvm -I\"${LIBCLANG_ROOT}/lib/clang/22/include\"")
set(OpenMP_C_FLAGS "/openmp:llvm -I\"${LIBCLANG_ROOT}/lib/clang/22/include\"")

# Force CMake to find libomp.lib
set(OpenMP_libomp_LIBRARY "${LLVM_LIB}/libomp.lib")
set(OpenMP_CXX_LIB_NAMES "libomp")
set(OpenMP_C_LIB_NAMES "libomp")

# Link against LLVM's libomp for OpenMP 5.1 support
# Use SHARED linker flags so it applies to DLLs too
set(CMAKE_SHARED_LINKER_FLAGS "/LIBPATH:\"${LLVM_LIB}\" libomp.lib")
set(CMAKE_EXE_LINKER_FLAGS "/LIBPATH:\"${LLVM_LIB}\" libomp.lib")
set(CMAKE_MODULE_LINKER_FLAGS "/LIBPATH:\"${LLVM_LIB}\" libomp.lib")

message(STATUS "Clang-CL toolchain loaded")
message(STATUS "  LLVM root: ${LIBCLANG_ROOT}")
message(STATUS "  OpenMP: LLVM libomp (5.1)")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Linker: ${CMAKE_LINKER}")
message(STATUS "  Windows SDK: ${WINSDK_ROOT} (${WINSDK_VERSION})")
