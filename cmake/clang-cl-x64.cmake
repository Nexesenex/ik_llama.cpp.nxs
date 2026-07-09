# Toolchain file for standalone LLVM/clang-cl x64 builds.
# This does NOT inherit the MSVC developer environment.
# All tool paths are set explicitly to C:\LLVM.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

set(CMAKE_C_COMPILER   "C:/LLVM/bin/clang-cl.exe")
set(CMAKE_CXX_COMPILER "C:/LLVM/bin/clang-cl.exe")

set(CMAKE_LINKER       "C:/LLVM/bin/lld-link.exe")
set(CMAKE_AR           "C:/LLVM/bin/llvm-ar.exe")
set(CMAKE_RANLIB       "C:/LLVM/bin/llvm-ranlib.exe")
set(CMAKE_RC_COMPILER  "C:/LLVM/bin/llvm-rc.exe")
set(CMAKE_MT           "C:/LLVM/bin/llvm-mt.exe")
set(CMAKE_DLLTOOL      "C:/LLVM/bin/llvm-dlltool.exe")

set(CMAKE_FIND_ROOT_PATH "C:/LLVM")

# Point at the MSVC standard library headers and libs from VS 2026
set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES
    "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/include"
    "P:/Windows Kits/10/Include/10.0.26100.0/ucrt"
    "P:/Windows Kits/10/Include/10.0.26100.0/shared"
    "P:/Windows Kits/10/Include/10.0.26100.0/um"
    "P:/Windows Kits/10/Include/10.0.26100.0/winrt"
)

set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_C_STANDARD_INCLUDE_DIRECTORIES})

# Point at the MSVC lib paths for linking
set(CMAKE_C_STANDARD_LIBRARIES
    "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/lib/x64"
    "P:/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64"
    "P:/Windows Kits/10/Lib/10.0.26100.0/um/x64"
)

set(CMAKE_CXX_STANDARD_LIBRARIES ${CMAKE_C_STANDARD_LIBRARIES})

# Use the LLVM OpenMP runtime
set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "")
set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "")
set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "")
set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "")
set(OpenMP_omp_LIBRARY "C:/LLVM/bin/libomp.dll" CACHE STRING "")

# Enable clang-specific target features
add_compile_options(-mavxvnni)

# Use lld for faster linking
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=lld")