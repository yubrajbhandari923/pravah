"""Compatibility alias: this module moved to pravah.studies.toy.windunc (same module object)."""
import sys

from .studies.toy import windunc as _m

sys.modules[__name__] = _m
