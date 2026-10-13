"""Compatibility alias: this module moved to pravah.studies.toy.experiments (same module object)."""
import sys

from .studies.toy import experiments as _m

sys.modules[__name__] = _m
