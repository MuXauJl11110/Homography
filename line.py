import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance


def get_lines(
    image: cv2.Mat,
    rho: Optional[float] = 1,
    theta: Optional[float] = np.pi / 180,
    threshold: Optional[int] = 200,
    lines: Optional[int] = 10,
    srn: Optional[int] = 15,
) -> List[List[Tuple[float, float]]]:
    # Converting to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blurring image to reduce noise.
    blur_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    # Generating Edge image
    edge_image = cv2.Canny(blur_gray_image, 40, 255)

    # Finding Lines in the image
    lines = cv2.HoughLines(edge_image, rho, theta, threshold, lines, srn)
    # Lines = cv2.HoughLinesP(EdgeImage, 1, np.pi / 180, 50, 10, 15)

    # Check if lines found and exit if not.
    if lines is None:
        print("get_lines: Not enough lines found in the image.")

    return lines


def filter_lines(
    lines: List[List[Tuple[float, float]]],
    angle: float,
    filter_num: Optional[int] = None,
    eps: Optional[float] = 0.1,
) -> List[List[Tuple[float, float]]]:
    filtered_lines = []
    for line in lines:
        [(_, theta)] = line
        if abs(theta - angle) < eps:
            filtered_lines.append(line)

    if filter_num is not None:
        return random.choices(filtered_lines, k=filter_num)

    return filtered_lines


def draw_line(
    image: cv2.Mat,
    line: List[Tuple[float, float]],
    color: Optional[Tuple[str, str, str]] = (0, 0, 255),
) -> cv2.Mat:
    [(rho, theta)] = line

    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho

    x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
    x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))

    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0, 0, 255) denotes the color of the line to be drawn.
    cv2.line(image, (x1, y1), (x2, y2), color, 2)

    return image


def get_homogeneous_coordinates(line: List[Tuple[float, float]]) -> List[float]:
    [(rho, theta)] = line
    return [np.cos(theta), np.sin(theta), -rho]


def get_intersection_point(line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]) -> List[float]:
    [a, b, c] = np.cross(get_homogeneous_coordinates(line1), get_homogeneous_coordinates(line2))

    return [a / c, b / c]


def get_intersection_line(line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]) -> List[float]:
    return np.cross(get_homogeneous_coordinates(line1), get_homogeneous_coordinates(line2))


def get_distance(point: Tuple[float, float], line: List[Tuple[float, float]]):
    [(rho, theta)] = line
    p_x, p_y = rho * np.cos(theta), rho * np.sin(theta)

    return abs(np.cos(theta) * (p_y - point[1]) - np.sin(theta) * (p_x - point[0]))


# https://stackoverflow.com/questions/15780210/python-opencv-detect-parallel-lines
def find_parallel_lines(
    lines: List[List[Tuple[float, float]]],
    filter_num: Optional[int] = None,
) -> List[List[List[Tuple[float, float]]]]:
    lines_ = lines[:, 0, :]
    angle = lines_[:, 1]

    # Perform hierarchical clustering

    angle_ = angle[..., np.newaxis]
    y = distance.pdist(angle_)
    Z = hierarchy.ward(y)
    cluster = hierarchy.fcluster(Z, 0.5, criterion="distance")

    parallel_lines = []
    for i in range(cluster.min(), cluster.max() + 1):
        temp = lines[np.where(cluster == i)]
        # Removing extra lines
        # (we might get many lines, so we are going to take only random filter_num lines
        # for further computation because more than this number of lines will only
        # contribute towards slowing down searching orthogonal lines.
        if filter_num is not None:
            parallel_lines.append(random.choices(temp, k=filter_num))
        else:
            parallel_lines.append(temp)

    return parallel_lines


def find_orthogonal_lines(
    clusters: Dict[int, List[List[Tuple[float, float]]]],
) -> List[List[List[Tuple[float, float]]]]:
    orthogonal_lines = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            candidates = []
            for line_i in clusters[i]:
                for line_j in clusters[j]:
                    [[_, theta1]], [[_, theta2]] = line_i, line_j
                    candidates.append(
                        (line_i, line_j, abs(np.cos(theta1) * np.cos(theta2) + np.sin(theta1) * np.sin(theta2)))
                    )

            candidates = sorted(candidates, key=lambda x: x[2])
            orthogonal_lines.append([candidates[0][0], candidates[0][1]])

    return orthogonal_lines
