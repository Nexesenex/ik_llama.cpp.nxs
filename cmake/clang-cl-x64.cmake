# Toolchain file for standalone LLVM/clang-cl x64 builds.
# This does NOT inherit the MSVC developer environment.
# All tool paths are set explicitly to C:\LLVM.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

set(CMAKE_C_COMPILER   "C:/LLVM/bin/clang-cl.exe")
set(CMAKE_CXX_COMPILER "C:/LLVM/bin/clang-cl.exe")

# Tell CMake clang-cl uses lld as its linker, not MSVC link.exe
set(CMAKE_LINKER_TYPE "LLD")

set(CMAKE_LINKER       "C:/LLVM/bin/lld-link.exe")
# Use llvm-lib (lib.exe-compatible) instead of llvm-ar for static libraries,
# because the Ninja generator passes MSVC-style flags like /nologo /machine:x64.
set(CMAKE_AR           "C:/LLVM/bin/llvm-lib.exe")
set(CMAKE_RANLIB       "C:/LLVM/bin/llvm-ranlib.exe")
set(CMAKE_RC_COMPILER  "C:/LLVM/bin/llvm-rc.exe")
set(CMAKE_MT           "C:/LLVM/bin/llvm-mt.exe")
set(CMAKE_DLLTOOL      "C:/LLVM/bin/llvm-dlltool.exe")

set(CMAKE_FIND_ROOT_PATH "C:/LLVM")

# Use the LLVM OpenMP runtime
set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "")
set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "")
set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "")
set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "")
set(OpenMP_omp_LIBRARY "C:/LLVM/lib/libomp.lib" CACHE STRING "")
set(OpenMP_omp_LIBRARY_DIR "C:/LLVM/lib" CACHE STRING "")

# Enable clang-specific target features
# Note: -mavxvnni enables avx2 as a dependency (defining __AVX2__) but NOT fma,
# so add -mfma explicitly to avoid clang's target-feature inlining errors.
add_compile_options(-mavxvnni -mfma)