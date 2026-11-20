# Optional build of the exact Clipper source used by the MCP OCR extension.
add_library(opencv_mcp_clipper STATIC clipper.cpp)
target_include_directories(opencv_mcp_clipper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(opencv_mcp_clipper PROPERTIES POSITION_INDEPENDENT_CODE ON)
