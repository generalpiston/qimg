from __future__ import annotations

import os

# Unit tests use the lightweight fallback embedder for speed and to avoid heavyweight runtime deps.
os.environ.setdefault("QIMG_ALLOW_FALLBACK", "1")
