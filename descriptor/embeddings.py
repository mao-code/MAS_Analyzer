from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA


def pca_2d(X: np.ndarray, *, random_state: int = 0) -> Tuple[np.ndarray, PCA]:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    pca = PCA(n_components=2, random_state=random_state)
    embedding = pca.fit_transform(X)
    return embedding, pca


def umap_2d(X: np.ndarray, *, random_state: int = 0, **kwargs) -> Tuple[np.ndarray, object]:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    try:
        import umap
    except Exception as exc:
        raise ImportError("umap-learn is not installed. Install it to use UMAP embeddings.") from exc
    reducer = umap.UMAP(n_components=2, random_state=random_state, **kwargs)
    embedding = reducer.fit_transform(X)
    return embedding, reducer
