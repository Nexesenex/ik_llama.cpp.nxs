# Toolchain file for standalone LLVM/clang-cl x64 builds.
# This does NOT inherit the MSVC developer environment.
# All tool paths are set explicitly to C:\LLVM.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

set(CMAKE_C_COMPILER   "C:/LLVM/bin/clang-cl.exe")
set(CMAKE_CXX_COMPILER "C:/LLVM/bin/clang-cl.exe")

set(CMAKE_LINKER_TYPE "LLD")

set(CMAKE_LINKER       "C:/LLVM/bin/lld-link.exe")
set(CMAKE_AR           "C:/LLVM/bin/llvm-ar.exe")
set(CMAKE_RANLIB       "C:/LLVM/bin/llvm-ranlib.exe")
set(CMAKE_RC_COMPILER  "C:/LLVM/bin/llvm-rc.exe")
set(CMAKE_MT           "C:/LLVM/bin/llvm-mt.exe")
set(CMAKE_DLLTOOL      "C:/LLVM/bin/llvm-dlltool.exe")

set(CMAKE_FIND_ROOT_PATH "C:/LLVM")

# Clang-cl -imsvc paths for system headers (no spaces in -imsvc flag)
set(MSVC_TOOLS_ROOT "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.52.36418")
set(WINKIT_ROOT "P:/Windows Kits/10")

set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES
    "${MSVC_TOOLS_ROOT}/include"
    "${WINKIT_ROOT}/Include/10.0.26100.0/ucrt"
    "${WINKIT_ROOT}/Include/10.0.26100.0/shared"
    "${WINKIT_ROOT}/Include/10.0.26100.0/um"
    "${WINKIT_ROOT}/Include/10.0.26100.0/winrt"
)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_C_STANDARD_INCLUDE_DIRECTORIES})

# Pass as -imsvc so clang-cl treats them as system include paths
foreach(dir ${CMAKE_C_STANDARD_INCLUDE_DIRECTORIES})
    list(APPEND CMAKE_C_FLAGS_INIT "-imsvc" "${dir}")
    list(APPEND CMAKE_CXX_FLAGS_INIT "-imsvc" "${dir}")
endforeach()

# Library search paths via /LIBPATH (lld-link handles quoted paths with spaces)
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "/LIBPATH:\"${MSVC_TOOLS_ROOT}/lib/x64\""
    "/LIBPATH:\"${WINKIT_ROOT}/Lib/10.0.26100.0/ucrt/x64\""
    "/LIBPATH:\"${WINKIT_ROOT}/Lib/10.0.26100.0/um/x64\""
)
set(CMAKE_SHARED_LINKER_FLAGS_INIT ${CMAKE_EXE_LINKER_FLAGS_INIT})
set(CMAKE_MODULE_LINKER_FLAGS_INIT ${CMAKE_EXE_LINKER_FLAGS_INIT})

# Use the LLVM OpenMP runtime
set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "")
set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "")
set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "")
set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "")
set(OpenMP_omp_LIBRARY "C:/LLVM/lib/libomp.lib" CACHE STRING "")
set(OpenMP_omp_LIBRARY_DIR "C:/LLVM/lib" CACHE STRING "")

# Enable clang-specific target features
add_compile_options(-mavxvnni)