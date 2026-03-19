# MSVC toolchain for x86_64-windows with Windows SDK paths

set(CMAKE_SYSTEM_NAME      Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/bin/Hostx64/x64/cl.exe")
set(CMAKE_RC_COMPILER  "P:/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe")
set(CMAKE_MT           "P:/Windows Kits/10/bin/10.0.26100.0/x64/mt.exe")
set(CUDA_NVCC_HOST_COMPILER "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/bin/Hostx64/x64/cl.exe")

# Set environment for both compiler and linker
set(ENV{LIB} "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/lib/x64;P:/Windows Kits/10/lib/10.0.26100.0/ucrt/x64;P:/Windows Kits/10/lib/10.0.26100.0/um/x64")
set(ENV{INCLUDE} "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/include;P:/Windows Kits/10/include/10.0.26100.0/ucrt;P:/Windows Kits/10/include/10.0.26100.0/um;P:/Windows Kits/10/include/10.0.26100.0/shared")
set(ENV{PATH} "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/bin/Hostx64/x64;P:/Windows Kits/10/bin/10.0.26100.0/x64;${env.PATH}")

# Set CMake paths for Windows SDK
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Add Windows SDK library path for linker - use Windows-style semicolons
set(CMAKE_EXE_LINKER_FLAGS "/LIBPATH:\"P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/lib/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/ucrt/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/um/x64\"")
set(CMAKE_SHARED_LINKER_FLAGS "/LIBPATH:\"P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/lib/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/ucrt/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/um/x64\"")
set(CMAKE_MODULE_LINKER_FLAGS "/LIBPATH:\"P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/lib/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/ucrt/x64\" /LIBPATH:\"P:/Windows Kits/10/lib/10.0.26100.0/um/x64\"")

# Add library search path
set(CMAKE_LIBRARY_PATH "P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/lib/x64;P:/Windows Kits/10/lib/10.0.26100.0/ucrt/x64;P:/Windows Kits/10/lib/10.0.26100.0/um/x64")

set(MSVC_C_FLAGS "/nologo /DWIN32 /D_WINDOWS /W3 /GR /EHsc /MP")

if(GGML_AVX2)
  set(MSVC_C_FLAGS "${MSVC_C_FLAGS} /arch:AVX2")
endif()
if(GGML_AVXVNNI)
  set(MSVC_C_FLAGS "${MSVC_C_FLAGS} /arch:AVX2 /D__AVXVNNI__=1")
endif()

set(CMAKE_C_FLAGS_INIT   "${MSVC_C_FLAGS} /I\"P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/include\" /I\"P:/Windows Kits/10/include/10.0.26100.0/ucrt\" /I\"P:/Windows Kits/10/include/10.0.26100.0/um\" /I\"P:/Windows Kits/10/include/10.0.26100.0/shared\" /I\"P:/Windows Kits/10/include/10.0.26100.0/winrt\"")
set(CMAKE_CXX_FLAGS_INIT "${MSVC_C_FLAGS} /I\"P:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/include\" /I\"P:/Windows Kits/10/include/10.0.26100.0/ucrt\" /I\"P:/Windows Kits/10/include/10.0.26100.0/um\" /I\"P:/Windows Kits/10/include/10.0.26100.0/shared\" /I\"P:/Windows Kits/10/include/10.0.26100.0/winrt\"")