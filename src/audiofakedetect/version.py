"""Version information."""

__all__ = [
    "VERSION",
    "get_version",
]

VERSION = "0.0.3-dev"


def get_version():
    """Get the version string."""
    return VERSION


if __name__ == "__main__":
    print(get_version())
