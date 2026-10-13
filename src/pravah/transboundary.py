"""Compatibility alias: this module moved to pravah.studies.toy.transboundary (same module object)."""
import sys

from .studies.toy import transboundary as _m

sys.modules[__name__] = _m
