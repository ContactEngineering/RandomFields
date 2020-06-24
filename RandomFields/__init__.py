
from . import Analysis, Generation  # noqa: F401

try:
    from importlib.metadata import version

    __version__ = version(__name__)
except ImportError:
    from pkg_resources import get_distribution

    __version__ = get_distribution(__name__).version
