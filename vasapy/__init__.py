from .dict import dict
from .set import set
__all__ = ["dict", "set"]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
