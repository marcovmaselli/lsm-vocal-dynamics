"""Logging helpers shared by experiment scripts.

The module provides a small tee-like stream duplicator so long-running training
and analysis scripts can mirror console output into a timestamped log file.
"""

import datetime
import os
import sys
from typing import TextIO


class Tee:
    """Mirror stdout and stderr into a log file.

    Instantiating this class has the side effect of replacing
    ``sys.stdout``/``sys.stderr`` globally with ``self`` until :meth:`close`
    is called (or the object is used as a context manager), so every
    ``print()`` in the process is duplicated to both the console and the
    log file. Only one ``Tee`` should be active at a time.
    """

    def __init__(self, filepath: str, mode: str = "w"):
        """Open ``filepath`` for writing and start mirroring stdout/stderr into it."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file: TextIO = open(filepath, mode, encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data: str) -> None:
        """Write ``data`` to both the log file and the original stdout."""
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self) -> None:
        self.file.flush()
        self.stdout.flush()

    def close(self) -> None:
        """Restore the original stdout/stderr and close the log file."""
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def __enter__(self) -> "Tee":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def setup_logger(log_dir: str, prefix: str = "run"):
    """Create a timestamped text log under ``log_dir`` and start mirroring stdout/stderr into it.

    Returns:
        Tuple of ``(log_path, tee)``: the path of the created log file, and
        the :class:`Tee` instance (keep a reference and call ``tee.close()``
        when done, or the redirect stays active for the rest of the process).
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.txt")

    tee_obj = Tee(log_path)

    print(f"[LOGGER] Writing to: {log_path}\n")
    return log_path, tee_obj