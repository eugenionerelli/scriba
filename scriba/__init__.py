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


def _quieten_torchcodec_warning() -> None:
    """Drop pyannote's twenty-line complaint that torchcodec cannot decode audio.

    It is correct and it does not apply here. pyannote 4 decodes through torchcodec,
    torchcodec needs FFmpeg 4 to 7 in the library path, and a pip install provides
    neither. scriba never reaches that code: audio is handed to pyannote as an
    in-memory waveform, which the warning itself lists as the supported way round it.

    Do not fix this by installing FFmpeg into the environment. That was tried.
    conda-forge's FFmpeg 7 lands in the environment's lib directory, PyAV already
    ships its own FFmpeg 8 under site-packages, and both get loaded into one process.
    macOS then reports duplicate Objective-C classes and warns about spurious casting
    failures and mysterious crashes. Trading a cosmetic warning for that, inside an
    audio pipeline, is a bad deal.

    If a future code path passes a file path to pyannote instead of a waveform, it
    will fail rather than warn. That is the cost, and it is the cheaper one.
    """
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*torchcodec is not installed correctly.*",
        category=UserWarning,
    )


_quieten_torchcodec_warning()

__version__ = "0.1.0"
