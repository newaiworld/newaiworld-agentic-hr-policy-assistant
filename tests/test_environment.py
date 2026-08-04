import sys


def test_python_version() -> None:
    """The project runtime is frozen at Python 3.11."""
    assert sys.version_info[:2] == (3, 11)