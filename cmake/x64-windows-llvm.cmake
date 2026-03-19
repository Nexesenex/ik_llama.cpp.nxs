# Base Clang→MSVC toolchain for x86_64-windows
# This file sets up the compiler and target triple, plus common tuning flags.

# Cross‑compile to Windows x86_64
set(CMAKE_SYSTEM_NAME      Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Use Clang from Visual Studio's LLVM toolchain
set(CMAKE_C_COMPILER   "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe")
set(CMAKE_CXX_COMPILER "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe")
set(CMAKE_RC_COMPILER  "P:/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe")
set(CMAKE_MT           "P:/Windows Kits/10/bin/10.0.26100.0/x64/mt.exe")
set(CUDA_NVCC_HOST_COMPILER "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/bin/Hostx64/x64/cl.exe")

# Add Windows SDK library path for linker
set(CMAKE_EXE_LINKER_FLAGS "/LIBPATH:\"P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/lib/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/ucrt/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/um/x64\"")
set(CMAKE_SHARED_LINKER_FLAGS "/LIBPATH:\"P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/lib/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/ucrt/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/um/x64\"")

# Set environment variables for Clang
set(ENV{LIB} "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/lib/x64;P:/Windows Kits/10/lib/10.0.26100.0/ucrt/x64;P:/Windows Kits/10/lib/10.0.26100.0/um/x64")
set(ENV{INCLUDE} "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/include;P:/Windows Kits/10/include/10.0.26100.0/ucrt;P:/Windows Kits/10/include/10.0.26100.0/um;P:/Windows Kits/10/include/10.0.26100.0/shared;P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/lib/clang/20/include")
set(ENV{PATH} "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin;P:/Windows Kits/10/bin/10.0.26100.0/x64;${env.PATH}")

# Common architecture tuning flags (no ISA extensions here)
# - Base x86_64 instruction set
# - Fast vectorization and FP model
# - Disable finite-math-only
set(arch_c_flags
  "-march=x86-64 \
   -fvectorize \
   -ffp-model=fast \
   -fno-finite-math-only"
)

# Warning suppression for this codebase
set(warn_c_flags
  "-Wno-format \
   -Wno-unused-variable \
   -Wno-unused-function \
   -Wno-gnu-zero-variadic-macro-arguments"
)

# Instruction set extensions based on GGML options
if(DEFINED GGML_AVX2)
  set(arch_c_flags "${arch_c_flags} -mavx2 -mfma -mf16c -mpopcnt")
if(DEFINED GGML_AVX512)
  set(arch_c_flags "${arch_c_flags} -mavx512f -mavx512vl -mavx512bw -mavx512dq")
endif()
if(DEFINED GGML_AVX512_BF16)
  set(arch_c_flags "${arch_c_flags} -mavx512bf16")
endif()
endif()

# Initialize flags for C and C++
set(CMAKE_C_FLAGS_INIT   "${arch_c_flags} ${warn_c_flags}")
set(CMAKE_CXX_FLAGS_INIT "${arch_c_flags} ${warn_c_flags}")
