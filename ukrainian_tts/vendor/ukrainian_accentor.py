"""
Ukrainian Accentor - included directly in the package to avoid PyPI dependency issues.
Original source: https://github.com/egorsmkv/ukrainian-accentor
"""

from torch.package import PackageImporter
from os.path import dirname, join
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Load the model from the included file
_model_path = join(dirname(__file__), "accentor-lite.pt")
_importer = PackageImporter(_model_path)
_accentor = _importer.load_pickle("uk-accentor", "model")

# Expose the process method
def process(text: str, mode: str = 'stress') -> str:
    """
    Add stress marks to Ukrainian text using the accentor model.
    
    Args:
        text: Ukrainian text without stress marks
        mode: Output mode ('stress' or 'plus')
        
    Returns:
        Ukrainian text with stress marks
    """
    return _accentor.process(text, mode)

__all__ = ["process"]
