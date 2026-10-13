"""Compatibility alias: this module moved to pravah.studies.belauri.belauri (same module object)."""
import sys

from .studies.belauri import belauri as _m

sys.modules[__name__] = _m
