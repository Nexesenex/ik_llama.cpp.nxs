include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/common/build-info.cpp.in")
set(OUTPUT_FILE   "${CMAKE_CURRENT_SOURCE_DIR}/common/build-info.cpp")

# Only write the build info if it changed
if(EXISTS ${OUTPUT_FILE})
    file(READ ${OUTPUT_FILE} CONTENTS)
    string(REGEX MATCH "LLAMA_COMMIT = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_COMMIT ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_COMPILER = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_COMPILER ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_TARGET = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_TARGET ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_BRANCH = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_BRANCH ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_DATE = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_DATE ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_LAST_MERGED_PR = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_LAST_MERGED_PR ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_CUDA_VERSION = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_CUDA_VERSION ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_LAST_COMMIT = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_LAST_COMMIT ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_GGML_CUDA = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_GGML_CUDA ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_NEXES_COMMITS = ([0-9]+);" _ ${CONTENTS})
    set(OLD_NEXES ${CMAKE_MATCH_1})
    if (
        NOT OLD_COMMIT         STREQUAL BUILD_COMMIT         OR
        NOT OLD_COMPILER       STREQUAL BUILD_COMPILER       OR
        NOT OLD_TARGET         STREQUAL BUILD_TARGET         OR
        NOT OLD_BRANCH         STREQUAL BUILD_BRANCH         OR
        NOT OLD_DATE           STREQUAL BUILD_DATE           OR
        NOT OLD_LAST_MERGED_PR STREQUAL BUILD_LAST_MERGED_PR OR
        NOT OLD_CUDA_VERSION   STREQUAL BUILD_CUDA_VERSION   OR
        NOT OLD_LAST_COMMIT   STREQUAL BUILD_LAST_COMMIT    OR
        NOT OLD_GGML_CUDA     STREQUAL BUILD_GGML_CUDA       OR
        NOT OLD_NEXES         STREQUAL BUILD_NEXES_COMMITS
    )
        configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})
    endif()
else()
    configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})
endif()
