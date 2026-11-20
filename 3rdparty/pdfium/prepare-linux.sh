#!/usr/bin/env bash
# Prepare a fresh Linux x86-64 PDFium checkout. Downloads upstream source/tools.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: bash prepare-linux.sh /absolute/path/to/new-work-directory" >&2
  exit 2
fi
work=$1
case "$work" in /*) ;; *) echo "Use an absolute work directory" >&2; exit 2;; esac
if [ -e "$work" ]; then
  echo "Refusing to reuse an existing work directory: $work" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git "$work/depot_tools"
git -C "$work/depot_tools" checkout e923de63bfc631d5a09ef5324635d008d50dc0ca
export PATH="$work/depot_tools:$PATH"
export DEPOT_TOOLS_UPDATE=0
cd "$work"
cat > .gclient <<'EOF'
solutions = [{"name": "pdfium", "url": "https://pdfium.googlesource.com/pdfium.git", "deps_file": "DEPS", "managed": False, "custom_deps": {}, "custom_vars": {"checkout_configuration": "minimal"}}]
target_os = ["linux"]
EOF
gclient sync --no-history --shallow --nohooks --revision pdfium@8ca5b735df4263f43c830479b78213854a392c22 --jobs 4
cd "$work/pdfium"
python3 tools/clang/scripts/update.py
python3 build/linux/sysroot_scripts/install-sysroot.py --arch=amd64
git apply "$script_dir/patches/shared_library.patch"
git apply "$script_dir/patches/public_headers.patch"
git -C build apply "$script_dir/patches/clang_rt.patch"
echo "Prepared PDFium source: $work/pdfium"
