from __future__ import annotations

import logging
import os


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def use_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("CLICOLOR") == "0":
        return False
    force = os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE")
    if force and force != "0":
        return True
    return True
