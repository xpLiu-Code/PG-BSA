"""PG-BSA inference API."""
from .api import TITLE, VERSION, create_model, verify_release

__version__ = VERSION
__all__ = ["TITLE", "create_model", "verify_release", "__version__"]
