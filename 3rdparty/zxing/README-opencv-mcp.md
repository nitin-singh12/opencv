# ZXing-C++ 3.1.1 for MCP

Source is from https://github.com/zxing-cpp/zxing-cpp at commit
`287c85df6f961c8efbfb5ffd736cd9457b8b890e`. The application pins each file's
hash in `apps/mcp/native/zxing-source-manifest.json`. Decoder support is built
with writers disabled, for the QR and barcode reader tools.

The OpenCV `BUILD_MCP_ZXING` option is off by default; it creates an optional
native decoder target and requires a C++20 compiler. MCP application wheels
build the Python binding separately from the same source and require the
nanobind build tool. The CMake target alone does not install that Python binding.
The upstream license is in LICENSE.
