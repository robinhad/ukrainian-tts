"""Deterministic Ukrainian text frontend used by training and inference."""

from .sanitize import SANITIZER_VERSION, sanitize_text

__all__ = ["SANITIZER_VERSION", "sanitize_text"]
