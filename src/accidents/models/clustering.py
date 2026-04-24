"""Embedding utilities for dimensionality reduction and visualization.

This module exposes a small set of methods to compute low-dimensional
embeddings and to visualize the results in a 3D scatter plot. The
implementation is designed to be reusable in notebooks and future model
selection workflows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS
from sklearn.metrics import pairwise_distances
import plotly.express as px


EMBEDDING_METHODS = ("pca", "mds", "isomap")


def fit_transform_embedding(
    X: pd.DataFrame | np.ndarray,
    method: str = "pca",
    n_components: int = 3,
    **kwargs: Any,
) -> np.ndarray:
    """Compute an embedding using one of the supported methods.

    This function dispatches to the appropriate embedding method based on
    the 'method' parameter. For sklearn-based methods (PCA, MDS, Isomap),
    it directly instantiates and fits the model. For custom methods like
    diffusion maps, it calls the dedicated function.
    """
    method_name = method.lower()

    if method_name == "pca":
        model = PCA(n_components=n_components, **kwargs)
    elif method_name == "mds":
        model = MDS(n_components=n_components, n_jobs=-1, **kwargs)
    elif method_name == "isomap":
        model = Isomap(n_components=n_components, **kwargs)
    else:
        raise ValueError(
            f"Unsupported embedding method '{method}'. "
            f"Supported methods: {', '.join(EMBEDDING_METHODS)}."
        )
    
    return model.fit_transform(X)


def plot_3d(
    embedding: np.ndarray,
    labels: pd.Series | np.ndarray,
    method_name: str,
    label_name: str = "GRAVITE_3",
) -> px.scatter_3d:
    """Create a 3D scatter plot for an embedding using Plotly.

    This function generates a consistent 3D visualization of the embedding
    results, colored by the provided labels (e.g., GRAVITE_3 categories).
    """
    if embedding.ndim != 2 or embedding.shape[1] != 3:
        raise ValueError("Embedding must be a 2D array with exactly 3 components.")

    df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2', 'Dim3'])
    df[label_name] = labels

    title = f"{method_name.upper()} 3D embedding ({label_name})"
    fig = px.scatter_3d(
        df,
        x="Dim1",
        y="Dim2",
        z="Dim3",
        color=label_name,
        title=title,
        opacity=0.8,
        width=900,
        height=700
    )
    fig.update_traces(marker={"size": 3})
    return fig


def compute_methods(
    X: pd.DataFrame | np.ndarray,
    methods: list[str] | None = None,
    n_components: int = 3,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Compute embeddings for multiple supported methods.

    This is a convenience function to compute embeddings for several methods
    in one call, useful for comparative analysis in notebooks.
    """
    methods = methods or list(EMBEDDING_METHODS)
    return {
        method: fit_transform_embedding(
            X,
            method=method,
            n_components=n_components,
            **kwargs,
        )
        for method in methods
    }

