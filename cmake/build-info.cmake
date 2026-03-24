set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")
set(BUILD_BRANCH "unknown")
set(BUILD_DATE "unknown")
set(BUILD_LAST_MERGED_PR "unknown")
set(BUILD_CUDA_VERSION "unknown")

# Look for git
find_package(Git)
if(NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if(GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
        message(STATUS "Found Git: ${GIT_EXECUTABLE}")
    else()
        message(WARNING "Git not found. Build info will not be accurate.")
    endif()
endif()

# Get the commit count and hash
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_COMMIT ${HEAD})
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE COUNT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_NUMBER ${COUNT})
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_BRANCH ${BRANCH})
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-list --count --author=Nexesenex HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE NEXES_COUNT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_NEXES_COMMITS ${NEXES_COUNT})
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log --all --oneline -100
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE RECENT_COMMITS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(RECENT_COMMITS)
        string(REGEX MATCHALL "#[0-9]+" PR_NUMS "${RECENT_COMMITS}")
        list(GET PR_NUMS 0 BUILD_LAST_MERGED_PR)
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log --oneline -1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE BUILD_LAST_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REPLACE "\"" "" BUILD_LAST_COMMIT_ESCAPED "${BUILD_LAST_COMMIT}")
endif()

string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M:%S" UTC)

if(WIN32)
    set(NVCC_EXE "P:/NVIDIAGPUCT/CUDA/v12.9/bin/nvcc.exe")
else()
    set(NVCC_EXE "nvcc")
endif()
if(EXISTS ${NVCC_EXE})
    execute_process(
        COMMAND ${NVCC_EXE} --version
        OUTPUT_VARIABLE NVCC_OUT
        RESULT_VARIABLE NVCC_RES
    )
    if(NVCC_RES EQUAL 0)
        string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _ ${NVCC_OUT})
        if(CMAKE_MATCH_1)
            set(BUILD_CUDA_VERSION ${CMAKE_MATCH_1})
        endif()
    endif()
endif()

if(MSVC)
    set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
else()
    execute_process(
        COMMAND sh -c "$@ --version | head -1" _ ${CMAKE_C_COMPILER}
        OUTPUT_VARIABLE OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(BUILD_COMPILER ${OUT})
    execute_process(
        COMMAND ${CMAKE_C_COMPILER} -dumpmachine
        OUTPUT_VARIABLE OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(BUILD_TARGET ${OUT})
endif()

if(BUILD_CUDA_VERSION)
    set(BUILD_GGML_CUDA "CUDA")
else()
    set(BUILD_GGML_CUDA "GGML_CUDA is not built")
endif()
