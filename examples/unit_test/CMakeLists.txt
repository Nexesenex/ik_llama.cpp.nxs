cmake_minimum_required(VERSION 3.14)
project(unit_test C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unit_test
    ../unit_test.cpp
    ../unit_test/fp16_table.cpp
    ../unit_test/stubs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/iqk/iqk_gemm_kquants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/iqk/iqk_gemm_legacy_quants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/iqk/iqk_quantize.cpp
)

target_include_directories(unit_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)

target_compile_definitions(unit_test PRIVATE
    _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    GGML_USE_IQK_MULMAT
    IQK_IMPLEMENT
)

if (MSVC)
    target_compile_options(unit_test PRIVATE /arch:AVX2)
    target_compile_definitions(unit_test PRIVATE __AVXVNNI__ __AVXVNNIINT8__)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(unit_test PRIVATE "-mavxvnni" "-mavxvnniint8")
    endif()
else()
    target_compile_options(unit_test PRIVATE -mavx2 -mavxvnni -mavxvnniint8)
endif()

# The fixed converter + reference quantizers are compiled directly into this
# executable (with IQK_IMPLEMENT). We do not link the ggml/llama DLLs.
target_link_libraries(unit_test PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# Differential VNNI battery: identical sources built WITH vs WITHOUT the
# VNNI fast paths. Both exes run the same fixed battery over identical
# random bytes and dump raw GEMM outputs; outputs must be bit-identical
# (integer dot products + identical float tails). Any differing byte is a
# behavioral change from the guard flips, not quantization noise.
set(DUMP_GEMM_SRCS
    ../unit_test/dump_kquants_vnni.cpp
    ../unit_test/stubs.cpp
    ../unit_test/fp16_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/iqk/iqk_gemm_kquants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/iqk/iqk_quantize.cpp
)

function(dump_gemm_target name vnni)
    add_executable(${name} ${DUMP_GEMM_SRCS})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    )
    target_compile_definitions(${name} PRIVATE
        _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
        GGML_USE_IQK_MULMAT
        IQK_IMPLEMENT
    )
    if (${vnni})
        if (MSVC)
            target_compile_options(${name} PRIVATE /arch:AVX2)
            target_compile_definitions(${name} PRIVATE __AVXVNNI__ __AVXVNNIINT8__)
            if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                target_compile_options(${name} PRIVATE "-mavxvnni" "-mavxvnniint8")
            endif()
        else()
            target_compile_options(${name} PRIVATE -mavx2 -mavxvnni -mavxvnniint8)
        endif()
    else()
        # Fallback path: plain AVX2, no VNNI macros (mirrors pre-flip codegen).
        if (MSVC)
            target_compile_options(${name} PRIVATE /arch:AVX2)
        else()
            target_compile_options(${name} PRIVATE -mavx2)
        endif()
    endif()
    target_link_libraries(${name} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endfunction()

dump_gemm_target(dump_kquants_vnni TRUE)
dump_gemm_target(dump_kquants_fallback FALSE)
