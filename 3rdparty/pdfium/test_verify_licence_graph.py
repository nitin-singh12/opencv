"""Check that unknown linked dependencies cannot bypass notice mapping."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('verify_licence_graph', Path(__file__).with_name('verify-licence-graph.py'))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class LicenceGraphTest(unittest.TestCase):
    def test_known_and_unknown_targets(self):
        with patch.object(verifier.subprocess, 'check_output', return_value='//third_party/harfbuzz:harfbuzz\n'):
            verifier.verify('gn', 'build')
        with patch.object(verifier.subprocess, 'check_output', return_value='//third_party/new_library:library\n'):
            with self.assertRaisesRegex(RuntimeError, 'Unmapped PDFium third-party target'):
                verifier.verify('gn', 'build')

    def test_compiler_runtime_and_dynamic_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            (build / 'obj').mkdir()
            link = build / 'obj/pdfium.ninja'
            link.write_text('  libs = /clang/libclang_rt.builtins.a -ldl -lpthread\n')
            output = ' 0x01 (NEEDED) Shared library: [libc.so.6]\n'
            with patch.object(verifier.subprocess, 'check_output', return_value=output):
                verifier.verify_link_inputs(build)
                link.write_text('  libs = /clang/libunknown.a -ldl\n')
                with self.assertRaisesRegex(RuntimeError, 'Unmapped compiler-runtime'):
                    verifier.verify_link_inputs(build)


if __name__ == '__main__':
    unittest.main()
