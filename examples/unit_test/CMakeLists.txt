set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unit_test
    ../unit_test.cpp
    ../unit_test/fp16_table.cpp
    ../unit_test/stubs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/iqk/iqk_gemm_kquants.cpp
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
    HAVE_VNNI256
    TEST_VNNI256
)

if (MSVC)
    target_compile_options(unit_test PRIVATE /arch:AVX2)
endif()

# The fixed converter + reference quantizers are compiled directly into this
# executable (with IQK_IMPLEMENT). We do not link the ggml/llama DLLs.
target_link_libraries(unit_test PRIVATE ${CMAKE_THREAD_LIBS_INIT})
