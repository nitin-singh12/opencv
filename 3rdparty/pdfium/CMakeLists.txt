# Experimental opt-in source build. PDFium uses GN/Ninja, not CMake.
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" OR CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR "The PDFium source-build target currently supports only native Linux x86-64.")
endif()

set(MCP_PDFIUM_SOURCE_DIR "" CACHE PATH "Prepared PDFium 8076 checkout with pinned dependencies and patches")
set(MCP_PDFIUM_GN_BUILD_DIR "${CMAKE_BINARY_DIR}/3rdparty/pdfium-gn" CACHE PATH "PDFium GN output directory")
set(MCP_PDFIUM_BUILD_JOBS "4" CACHE STRING "Parallel PDFium compile jobs")
if(NOT IS_ABSOLUTE "${MCP_PDFIUM_SOURCE_DIR}" OR NOT EXISTS "${MCP_PDFIUM_SOURCE_DIR}/BUILD.gn")
  message(FATAL_ERROR "Set MCP_PDFIUM_SOURCE_DIR to an absolute path to the prepared PDFium checkout; see 3rdparty/pdfium/README.md.")
endif()
if(NOT MCP_PDFIUM_BUILD_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "MCP_PDFIUM_BUILD_JOBS must be a positive integer.")
endif()
set(MCP_PDFIUM_GN_EXECUTABLE "${MCP_PDFIUM_SOURCE_DIR}/buildtools/linux64/gn")
find_program(MCP_PDFIUM_NINJA_EXECUTABLE NAMES ninja)
if(NOT EXISTS "${MCP_PDFIUM_GN_EXECUTABLE}" OR NOT MCP_PDFIUM_NINJA_EXECUTABLE)
  message(FATAL_ERROR "The PDFium source-build target requires GN and Ninja.")
endif()
find_package(Git REQUIRED)
find_program(MCP_PDFIUM_PYTHON_EXECUTABLE NAMES python3)
if(NOT MCP_PDFIUM_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "The PDFium source-build target requires Python 3.")
endif()

# The checks run again at build time, so changes after CMake configuration fail closed.
set(_pdfium_check "${CMAKE_CURRENT_SOURCE_DIR}/verify-source.cmake")
execute_process(COMMAND "${CMAKE_COMMAND}" "-DSOURCE=${MCP_PDFIUM_SOURCE_DIR}" "-DGIT=${GIT_EXECUTABLE}" -P "${_pdfium_check}"
                RESULT_VARIABLE _pdfium_check_result)
if(NOT _pdfium_check_result EQUAL 0)
  message(FATAL_ERROR "The prepared PDFium checkout failed verification.")
endif()
set(_pdfium_dependencies_check "${CMAKE_CURRENT_SOURCE_DIR}/verify-dependencies.py")
execute_process(COMMAND "${MCP_PDFIUM_PYTHON_EXECUTABLE}" "${_pdfium_dependencies_check}" "${MCP_PDFIUM_SOURCE_DIR}" "${GIT_EXECUTABLE}"
                RESULT_VARIABLE _pdfium_dependencies_result)
if(NOT _pdfium_dependencies_result EQUAL 0)
  message(FATAL_ERROR "The PDFium dependency check failed.")
endif()
set(_pdfium_args "is_debug=false pdf_is_standalone=true pdf_use_partition_alloc=false target_cpu=\"x64\" target_os=\"linux\" pdf_enable_v8=false pdf_enable_xfa=false treat_warnings_as_errors=false is_component_build=false clang_use_chrome_plugins=false use_remoteexec=false pdf_use_skia=false")
add_custom_target(opencv_mcp_pdfium
  COMMAND "${CMAKE_COMMAND}" "-DSOURCE=${MCP_PDFIUM_SOURCE_DIR}" "-DGIT=${GIT_EXECUTABLE}" -P "${_pdfium_check}"
  COMMAND "${MCP_PDFIUM_PYTHON_EXECUTABLE}" "${_pdfium_dependencies_check}" "${MCP_PDFIUM_SOURCE_DIR}" "${GIT_EXECUTABLE}"
  COMMAND "${MCP_PDFIUM_GN_EXECUTABLE}" gen "${MCP_PDFIUM_GN_BUILD_DIR}" "--args=${_pdfium_args}"
  COMMAND "${MCP_PDFIUM_NINJA_EXECUTABLE}" -C "${MCP_PDFIUM_GN_BUILD_DIR}" -j "${MCP_PDFIUM_BUILD_JOBS}" pdfium
  COMMAND "${MCP_PDFIUM_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/verify-licence-graph.py" "${MCP_PDFIUM_GN_EXECUTABLE}" "${MCP_PDFIUM_GN_BUILD_DIR}"
  WORKING_DIRECTORY "${MCP_PDFIUM_SOURCE_DIR}"
  BYPRODUCTS "${MCP_PDFIUM_GN_BUILD_DIR}/libpdfium.so"
  COMMENT "Building pinned PDFium 8076 with GN/Ninja (no V8 or XFA)"
  VERBATIM)
