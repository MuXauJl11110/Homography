from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

# from line import get_distance, get_intersection_point

# def get_vanishing_point(lines: List[List[Tuple[float, float]]]):
#     # We will apply RANSAC inspired algorithm for this. We will take combination
#     # of 2 lines one by one, find their intersection point, and calculate the
#     # total error(loss) of that point. Error of the point means root of sum of
#     # squares of distance of that point from each line.
#     vanishing_point = None
#     min_error = float("inf")

#     for i in range(len(lines)):
#         for j in range(i + 1, len(lines)):
#             [(_, theta1)], [(_, theta2)] = lines[i], lines[j]
#             x0, y0 = get_intersection_point(lines[i], lines[j])

#             if theta1 != theta2:
#                 err = 0
#                 for line in lines:
#                     err += get_distance((x0, y0), line)

#                 if err < min_error:
#                     min_error = err
#                     vanishing_point = [x0, y0]

#     return vanishing_point


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
