"""Logging helpers shared by experiment scripts.

The module provides a small tee-like stream duplicator so long-running training
and analysis scripts can mirror console output into a timestamped log file.
"""

import datetime
import os
import sys
from typing import TextIO


class Tee:
    """Mirror stdout and stderr into a log file."""

    def __init__(self, filepath: str, mode: str = "w"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file: TextIO = open(filepath, mode, encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data: str) -> None:
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self) -> None:
        self.file.flush()
        self.stdout.flush()

    def close(self) -> None:
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def __enter__(self) -> "Tee":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def setup_logger(log_dir: str, prefix: str = "run"):
    """Create a timestamped text log and start mirroring stdout/stderr."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.txt")

    tee_obj = Tee(log_path)

    print(f"[LOGGER] Writing to: {log_path}\n")
    return log_path, tee_obj