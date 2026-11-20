"""Private native components built with the package."""

ZXING_VERSION = "3.1.1"


def load_zxing():
    from . import zxingcpp
    return zxingcpp
