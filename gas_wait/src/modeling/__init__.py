"""Gas Wait modeling package — features, backtesting, and signals."""

from .config import FeatureGroup, resolve_feature_columns
from .signals import SignalResult, generate_signal

__all__ = [
    "FeatureGroup",
    "SignalResult",
    "generate_signal",
    "resolve_feature_columns",
]
