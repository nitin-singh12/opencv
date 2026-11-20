"""Check wheel labels without requiring a macOS host."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


def wheel_class(system, machine, tag):
    class Base:
        def get_tag(self):
            return 'cp312', 'cp312', tag

        def run(self):
            return 'built'

    path = Path(__file__).resolve().parents[1] / 'setup.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeWheel')
    namespace = {'bdist_wheel': Base, 'sys': SimpleNamespace(platform=system),
                 'platform': SimpleNamespace(machine=lambda: machine), 're': re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['NativeWheel']()


@pytest.mark.parametrize(('original', 'expected'), [
    ('macosx_10_13_universal2', 'macosx_13_0_arm64'),
    ('macosx_11_0_arm64', 'macosx_13_0_arm64'),
    ('macosx_15_0_universal2', 'macosx_15_0_arm64'),
    ('macosx_14_2_arm64', 'macosx_14_2_arm64'),
])
def test_macos_arm_wheel_is_narrowed_and_respects_pdfium_minimum(original, expected):
    wheel = wheel_class('darwin', 'arm64', original)
    assert wheel.get_tag() == ('cp312', 'cp312', expected)
    assert wheel.run() is None


@pytest.mark.parametrize(('system', 'machine', 'tag'), [
    ('darwin', 'x86_64', 'macosx_13_0_universal2'),
    ('darwin', 'arm64', 'macosx_13_0_x86_64'),
    ('darwin', 'arm64', 'macosx_bad_universal2'),
    ('linux', 'x86_64', 'macosx_13_0_universal2'),
    ('win32', 'amd64', 'macosx_13_0_universal2'),
])
def test_incompatible_wheel_platform_is_rejected(system, machine, tag):
    wheel = wheel_class(system, machine, tag)
    assert wheel.get_tag() == ('cp312', 'cp312', tag)
    with pytest.raises(RuntimeError, match='does not match'):
        wheel.run()


def test_supported_linux_and_windows_tags_are_unchanged():
    for system, machine, tag in [('linux', 'x86_64', 'linux_x86_64'),
                                 ('win32', 'amd64', 'win_amd64')]:
        wheel = wheel_class(system, machine, tag)
        assert wheel.get_tag() == ('cp312', 'cp312', tag)
        assert wheel.run() is None


@pytest.mark.parametrize(('output', 'accepted'), [
    ('Load command 1\n cmd LC_BUILD_VERSION\n minos 13.0\n', True),
    ('Load command 1\n cmd LC_VERSION_MIN_MACOSX\n version 12.3\n', True),
    ('Load command 1\n cmd LC_BUILD_VERSION\n minos 14.0\n', False),
    ('Load command 1\n cmd LC_SOURCE_VERSION\n version 1.0\n', False),
])
def test_macos_native_binary_minimum_must_fit_wheel(tmp_path, output, accepted):
    path = Path(__file__).resolve().parents[1] / 'setup.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeBuild')
    commands = []
    subprocess = SimpleNamespace(run=lambda args, check: commands.append(args),
                                 check_output=lambda *args, **kwargs: output)
    namespace = {'build_ext': object, 'Path': Path, 'root': tmp_path, 're': re,
                 'subprocess': subprocess}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    wheel = SimpleNamespace(ensure_finalized=lambda: None,
                            get_tag=lambda: ('cp312', 'cp312', 'macosx_13_0_arm64'))
    build = namespace['NativeBuild']()
    build.distribution = SimpleNamespace(get_command_obj=lambda name: wheel)
    build.inplace = False
    build.build_lib = str(tmp_path)
    build.extensions = [SimpleNamespace(name='clipper'), SimpleNamespace(name='zxing')]
    build.get_ext_fullpath = lambda name: str(tmp_path / name)
    if accepted:
        build.verify_macos_binaries()
        assert len(commands) == 3
        assert all(command[1:3] == ['-verify_arch', 'arm64'] for command in commands)
    else:
        with pytest.raises(RuntimeError, match='deployment version'):
            build.verify_macos_binaries()
