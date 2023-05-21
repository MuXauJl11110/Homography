from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression


# https://stackoverflow.com/questions/57535865/extract-vanishing-point-from-lines-with-open-cv
def get_vanishing_point(lines: List[List[Tuple[float, float]]]):
    rhos, thetas = [], []

    for line in lines:
        [(rho, theta)] = line
        rhos.append(rho)
        thetas.append([np.cos(theta), np.sin(theta)])

    reg = LinearRegression().fit(thetas, rhos)

    return reg.coef_


def get_distant_points(vanishing_points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], float]:
    two_points = []
    max_dist = 0

    for i in range(len(vanishing_points)):
        for j in range(i + 1, len(vanishing_points)):
            vp1, vp2 = vanishing_points[i], vanishing_points[j]
            dist = np.sqrt((vp1[0] - vp2[0]) ** 2 + (vp1[1] - vp2[1]) ** 2)
            if dist > max_dist:
                max_dist = dist
                two_points = [vp1, vp2]

    return two_points, max_dist


def get_nearest_points(vanishing_points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], float]:
    two_points = []
    min_dist = float("inf")

    for i in range(len(vanishing_points)):
        for j in range(i + 1, len(vanishing_points)):
            vp1, vp2 = vanishing_points[i], vanishing_points[j]
            dist = np.sqrt((vp1[0] - vp2[0]) ** 2 + (vp1[1] - vp2[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                two_points = [vp1, vp2]

    return two_points, min_dist
