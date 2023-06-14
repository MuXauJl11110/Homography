from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.transform import ProjectiveTransform
from sklearn.linear_model import LinearRegression

from line import (
    get_homogeneous_coordinates,
    get_intersection_point,
    get_kb,
    get_lines_skimage,
)


# https://stackoverflow.com/questions/57535865/extract-vanishing-point-from-lines-with-open-cv
def get_vanishing_point_LR(lines: List[List[Tuple[float, float]]]):
    rhos, thetas = [], []

    for line in lines:
        [(rho, theta)] = line
        rhos.append(rho)
        thetas.append([np.cos(theta), np.sin(theta)])

    reg = LinearRegression().fit(thetas, rhos)

    return reg.coef_


# https://github.com/knowindeep-int/VanishingPoint/blob/master/main.py
def get_vanishing_point(lines: List[List[Tuple[float, float]]]):
    # We will apply RANSAC inspired algorithm for this. We will take combination
    # of 2 lines one by one, find their intersection point, and calculate the
    # total error(loss) of that point. Error of the point means root of sum of
    # squares of distance of that point from each line.
    vanishing_point = None
    min_error = 100000000000

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            [(theta_i, rho_i)] = lines[i]
            [(theta_j, rho_j)] = lines[j]
            m1, c1 = get_kb(theta_i, rho_i)
            m2, c2 = get_kb(theta_j, rho_j)

            if m1 != m2:
                x0 = (c1 - c2) / (m2 - m1)
                y0 = m1 * x0 + c1

                err = 0
                for k in range(len(lines)):
                    [(theta_k, rho_k)] = lines[k]
                    m, c = get_kb(theta_k, rho_k)
                    m_ = -1 / m
                    c_ = y0 - m_ * x0

                    x_ = (c - c_) / (m_ - m)
                    y_ = m_ * x_ + c_

                    l = np.sqrt((y_ - y0) ** 2 + (x_ - x0) ** 2)

                    err += l**2

                err = np.sqrt(err)

                if min_error > err:
                    min_error = err
                    vanishing_point = [x0, y0]

    return vanishing_point


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


def get_nearest_points(
    vanishing_points: List[Tuple[float, float]], ret_indexes: Optional[bool] = False
) -> Tuple[List[Tuple[float, float]], float]:
    two_points = []
    min_dist = float("inf")

    for i in range(len(vanishing_points)):
        for j in range(i + 1, len(vanishing_points)):
            vp1, vp2 = vanishing_points[i], vanishing_points[j]
            dist = np.sqrt((vp1[0] - vp2[0]) ** 2 + (vp1[1] - vp2[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                two_points = [i, j] if ret_indexes else [vp1, vp2]

    return two_points, min_dist
