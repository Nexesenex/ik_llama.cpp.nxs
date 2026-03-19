# MSVC + LLVM OpenMP toolchain for llama.cpp
# This uses MSVC for compilation and LLVM's libomp for OpenMP 5.1 support
# Usage: cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/msvc-llvm-omp.cmake -B build -S .
# Requires: LLVM 22+ installed at C:/Program Files/LLVM

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# LLVM/Clang installation (for OpenMP runtime only)
set(LIBCLANG_ROOT "C:/Program Files/LLVM" CACHE PATH "LLVM installation directory")

# MSVC 2026 installation
set(MSVC_ROOT "P:/Program Files/Microsoft Visual Studio/18/Community")
set(MSVC_VERSION "14.51.36014")

# Windows SDK
set(WINSDK_ROOT "P:/Windows Kits/10")
set(WINSDK_VERSION "10.0.26100.0")

# Use MSVC as the compiler (for better Windows header compatibility)
set(CMAKE_C_COMPILER   "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/cl.exe")

# Resource compiler from MSVC
set(CMAKE_RC_COMPILER "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/bin/Hostx64/x64/rc.exe")

# Architecture
set(CMAKE_VC_COMPILER_ID MSVC)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include and lib paths
set(MSVC_INCLUDE    "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/include")
set(MSVC_LIB        "${MSVC_ROOT}/VC/Tools/MSVC/${MSVC_VERSION}/lib/x64")
set(WINSDK_INC      "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/ucrt;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/shared;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/um")
set(WINSDK_LIB      "${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/ucrt/x64;${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/um/x64")

# Common include paths
include_directories(
    "${MSVC_INCLUDE}"
    ${WINSDK_INC}
)

# Common library paths
link_directories(
    "${MSVC_LIB}"
    ${WINSDK_LIB}
    "${LIBCLANG_ROOT}/lib"
)

# Link against LLVM's libomp for OpenMP 5.1 support
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LIBPATH:\"${LIBCLANG_ROOT}/lib\" libomp.lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LIBPATH:\"${LIBCLANG_ROOT}/lib\" libomp.lib")

message(STATUS "MSVC + LLVM OpenMP toolchain loaded")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  OpenMP: LLVM libomp (5.1) from ${LIBCLANG_ROOT}")
message(STATUS "  Windows SDK: ${WINSDK_ROOT} (${WINSDK_VERSION})")
