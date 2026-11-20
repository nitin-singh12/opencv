"""Declare PDFium's binary provider only for prebuilt-wheel builds."""
import os
from setuptools import build_meta as backend

build_wheel = backend.build_wheel
build_sdist = backend.build_sdist
prepare_metadata_for_build_wheel = backend.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = backend.get_requires_for_build_sdist
build_editable = backend.build_editable
prepare_metadata_for_build_editable = backend.prepare_metadata_for_build_editable


def get_requires_for_build_wheel(config_settings=None):
    requirements = backend.get_requires_for_build_wheel(config_settings)
    if not os.environ.get('OPENCV_MCP_PDFIUM_SOURCE_LIBRARY'):
        requirements.append('pypdfium2==5.14.0')
    return requirements


def get_requires_for_build_editable(config_settings=None):
    requirements = backend.get_requires_for_build_editable(config_settings)
    if not os.environ.get('OPENCV_MCP_PDFIUM_SOURCE_LIBRARY'):
        requirements.append('pypdfium2==5.14.0')
    return requirements
