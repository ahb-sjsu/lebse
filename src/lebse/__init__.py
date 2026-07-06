"""LeBSE — legal-domain sentence embeddings adapted from LaBSE.

`import lebse` is intentionally light (no torch). The heavy encoder is loaded lazily via
`lebse.load(...)`, which imports `sentence_transformers` only when called.
"""

from __future__ import annotations

__version__ = "0.1.0"

from lebse.model import DEFAULT_MODEL, load

__all__ = ["DEFAULT_MODEL", "load", "__version__"]
