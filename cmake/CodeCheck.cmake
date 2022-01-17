# additional target to perform clang-format run, requires clang-format

find_program(CLANG_FORMAT_EXECUTABLE
             NAMES clang-format
                   clang-format-7
                   clang-format-6.0
                   clang-format-5.0
                   clang-format-4.0
                   clang-format-3.9
                   clang-format-3.8
                   clang-format-3.7
                   clang-format-3.6
                   clang-format-3.5
                   clang-format-3.4
                   clang-format-3.3
             DOC "clang-format executable")
mark_as_advanced(CLANG_FORMAT_EXECUTABLE)

if(CLANG_FORMAT_EXECUTABLE)
    # get all project files
    file(GLOB_RECURSE ALL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/intel_extension_for_pytorch/csrc/*.cpp" "${PROJECT_SOURCE_DIR}/intel_extension_for_pytorch/csrc/*.h")
    foreach (SOURCE_FILE ${ALL_SOURCE_FILES})
        #message(${SOURCE_FILE})
        execute_process(COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${SOURCE_FILE} )
    endforeach ()
endif()


