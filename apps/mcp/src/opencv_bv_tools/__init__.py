"""Local image tools. Importing the package does not start an MCP server."""

__all__ = ["ConversionError", "ConversionResult", "convert_image"]


def __getattr__(name):
    if name in __all__:
        from . import image_conversion

        return getattr(image_conversion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
