"""Project-wide default configuration values.

The dictionary is intentionally flat because most experiment scripts pass these
values around as keyword arguments.
"""

CONFIG = {
    # Audio I/O
    "sr": 16000,
    "mono": True,
    "tier_name": "default",
    # Preprocessing
    "highcut": 7900,
    "min_silence_duration": 0.05,
    # Feature extraction
    "win_length_ms": 64,
    "hop_length_ms": 16,
    "log_compression": True,
    "n_mels": 128,
    "n_filter": 128,
}