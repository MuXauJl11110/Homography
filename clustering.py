import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from skimage.measure import LineModelND, ransac
from sklearn.cluster import KMeans

from line import draw_lines, get_kb


def cluster_lines(
    lines: List[List[Tuple[float, float]]], n_clusters: int
) -> Dict[int, List[List[Tuple[float, float]]]]:
    angels = [[np.cos(line[0][1]), np.sin(line[0][1])] for line in lines]

    km = KMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(angels)

    clusters = defaultdict(list)

    for i, line in enumerate(lines):
        clusters[y_km[i]].append(line)

    return clusters


def cluster_lines_skimage(
    thetas: np.ndarray, rhos: np.ndarray, n_clusters: int, plot: Optional[bool] = False
) -> Dict[int, List[List[Tuple[float, float]]]]:
    angels = [[np.cos(theta), np.sin(theta)] for theta in thetas]

    km = KMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(angels)

    clusters = defaultdict(list)

    for i, (theta, rho) in enumerate(zip(thetas, rhos)):
        clusters[y_km[i]].append([(theta, rho)])

    if plot:
        _, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax = ax.flatten()

        def get_cmap(n, name="nipy_spectral"):
            """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name."""
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(n_clusters)
        colors = [cmap(i) for i in y_km]
        ax[0].scatter(thetas, rhos, c=colors)
        ax[0].set_xlabel(r"$\theta$, degrees")
        ax[0].set_ylabel(r"$\rho$")
        ax[0].set_title(r"$x\,cos(\theta)+ y\,sin(\theta)=\rho$")

        k, b = zip(*[get_kb(theta, rho) for theta, rho in zip(thetas, rhos)])
        ax[1].scatter(k, b, c=colors)
        ax[1].set_xlabel("k")
        ax[1].set_ylabel("b")
        ax[1].set_title(r"$y = k\cdot x+b$")

        plt.show()

    return clusters


def cluster_ransac(
    thetas: np.ndarray,
    rhos: np.ndarray,
    min_samples: Optional[float] = 5,
    residual_threshold: Optional[float] = 0.05,
    plot: Optional[bool] = False,
) -> np.ndarray:
    data = np.array([get_kb(theta, rho) for theta, rho in zip(thetas, rhos)])

    # robustly fit line only using inlier data with RANSAC algorithm
    _, inliers = ransac(
        data,
        LineModelND,
        min_samples=min(min_samples, len(data) - 1),
        residual_threshold=residual_threshold,
        max_trials=1000,
    )

    if plot:
        _, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax = ax.flatten()

        colors = np.array(["green" if inl else "red" for inl in inliers])
        ax[0].scatter(thetas, rhos, c=colors)
        ax[0].set_xlabel(r"$\theta$, degrees")
        ax[0].set_ylabel(r"$\rho$")
        ax[0].set_title(r"$x\,cos(\theta)+ y\,sin(\theta)=\rho$")

        k, b = zip(*[get_kb(theta, rho) for theta, rho in zip(thetas, rhos)])
        ax[1].scatter(k, b, c=colors)
        ax[1].set_xlabel("k")
        ax[1].set_ylabel("b")
        ax[1].set_title(r"$y = k\cdot x+b$")

        plt.show()

    return inliers
