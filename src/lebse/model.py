"""Loading the LeBSE encoder.

The model is a standard `sentence-transformers` model, so anything that accepts a
SentenceTransformer works unchanged. Weights are distributed via the Hugging Face Hub (or a local
path); they are NOT shipped in the PyPI wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep `import lebse` torch-free; only for type checkers
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "ahbond/lebse"


def load(model: str = DEFAULT_MODEL, device: str | None = None) -> SentenceTransformer:
    """Load a LeBSE (or any sentence-transformers) model.

    Args:
        model: Hugging Face Hub id or a local directory containing the model.
        device: "cuda", "cpu", or None to let sentence-transformers choose.

    Returns:
        A ready-to-use ``SentenceTransformer``. Call ``.encode(texts, normalize_embeddings=True)``.

    Raises:
        ImportError: if the ``[train]``/``[eval]`` extra (sentence-transformers) is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "lebse.load requires sentence-transformers. Install with: pip install 'lebse[eval]'"
        ) from exc
    return SentenceTransformer(model, device=device)
