from __future__ import annotations

from pathlib import Path
import os

APP_NAME = "qimg"


def cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".cache"
    root = base / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def config_root() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".config"
    root = base / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_db_path() -> Path:
    return cache_root() / "qimg.sqlite3"


def default_vector_path() -> Path:
    return cache_root() / "vectors"


def default_pid_path() -> Path:
    return cache_root() / "mcp_http.pid"
