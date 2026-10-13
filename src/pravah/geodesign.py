"""Compatibility alias: this module moved to pravah.studies.belauri.geodesign (same module object)."""
import sys

from .studies.belauri import geodesign as _m

sys.modules[__name__] = _m
