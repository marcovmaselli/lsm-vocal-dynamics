"""Analysis helpers used by downstream clustering experiments."""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

__all__ = ["run_pca_kmeans"]


def run_pca_kmeans(
    subject_features: Dict[str, np.ndarray],
    n_components: int = 2,
    n_clusters: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run PCA and KMeans on subject-level feature vectors.

    Returns:
        X: stacked subject features with shape ``(n_subjects, n_features)``.
        X_2d: PCA projection with shape ``(n_subjects, n_components)``.
        clusters: cluster assignment for each subject.
        subjects_list: subject identifiers in the same order as ``X``.
    """
    subjects_list = list(subject_features.keys())
    if len(subjects_list) == 0:
        raise RuntimeError("Nessun soggetto processato: subject_features vuoto")

    X = np.stack([subject_features[s] for s in subjects_list], axis=0)
    pca = PCA(n_components=n_components)
    X_2d = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    print(f"[PCA] Varianza spiegata: {explained_var}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_2d)

    return X, X_2d, clusters, subjects_list
