"""scriba: from voice memo to NotebookLM source."""

import os as _os

# Turn pyannote's telemetry off before anything can import pyannote.
#
# pyannote 4 ships a telemetry module whose config.yaml sets metrics_enabled: true,
# reporting to an endpoint at pyannote.ai on every from_pretrained and every file
# processed. The opt-out is read once, guarded by
# `if "PYANNOTE_METRICS_ENABLED" not in os.environ`, so it only counts if it is set
# before the import. It sits here, in the package root, because this module runs
# first no matter which entry point is used. Putting it in diarize.py worked only
# because of the order imports happened to run in.
#
# These are recordings of private conversations. The default is not acceptable and
# the fix has to be one that cannot drift.
_os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

__version__ = "0.1.0"
