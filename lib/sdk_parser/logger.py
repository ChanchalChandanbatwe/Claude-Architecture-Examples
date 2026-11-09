import os
import sys
from datetime import datetime

# Capture timestamp once at import time so the whole run shares one log file.
# Colons are avoided so the name is valid on Windows.
_RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_log_file = None


def _project_dir() -> str:
    """Directory of the entry-point script (e.g. hello_world/), not the cwd.

    Falls back to the cwd for interactive sessions / `python -c`.
    """
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file:
        return os.path.dirname(os.path.abspath(main_file))
    return os.getcwd()


def _get_log_file():
    global _log_file
    if _log_file is not None:
        return _log_file

    log_dir = os.path.join(_project_dir(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{_RUN_TIMESTAMP}.log")
    _log_file = open(log_path, "a", encoding="utf-8")
    return _log_file


def write(text: str) -> None:
    f = _get_log_file()
    f.write(text + "\n")
    f.flush()
