"""
RND Extraction Pipeline - Re-export from src
"""

import sys
from pathlib import Path

# Add src path
_src_path = Path(__file__).parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Re-export everything from the actual package
from rnd_extraction.core import extract_rnd
from rnd_extraction.core_optimized import extract_rnd_optimized
from rnd_extraction.core_ultra import extract_rnd_ultra, extract_rnd_ultra_simple
from rnd_extraction.incremental import GPState
from rnd_extraction.types import GPHyperparameters, MarketData, RNDResult

__all__ = [
    "extract_rnd",
    "extract_rnd_optimized",
    "extract_rnd_ultra",
    "extract_rnd_ultra_simple",
    "MarketData",
    "RNDResult",
    "GPHyperparameters",
    "GPState",
]

__version__ = "2.0.0"


