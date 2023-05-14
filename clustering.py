from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans


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
