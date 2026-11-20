# Clipper 6.4.2

The source files here are the Clipper C++ sources from pyclipper 1.4.0,
commit `f998920369565dca56d346bb30c5741e273f0b9b` in
https://github.com/fonttools/pyclipper. The original source retains its
Boost Software License 1.0 notice. See LICENSE.txt.

OpenCV does not build this library by default. `BUILD_MCP_CLIPPER=ON`
provides an optional static CMake target for evaluation. The MCP Python wheel
compiles the same pinned source into its own extension; enabling the CMake
option alone does not install the Python extension.
