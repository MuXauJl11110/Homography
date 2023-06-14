import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from skimage.transform import hough_line, hough_line_peaks


def get_lines(
    image: cv2.Mat,
    rho: Optional[float] = 1,
    theta: Optional[float] = np.pi / 180,
    threshold: Optional[int] = 200,
    lines: Optional[int] = 100,
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


def get_lines_skimage(
    image: np.ndarray,
    angle_from: Optional[float] = -np.pi / 4,
    angle_to: Optional[float] = np.pi / 4,
) -> Tuple[np.ndarray, np.ndarray]:
    # Finding Lines in the image
    hspace_all, theta_all, rho_all = hough_line(image, theta=np.linspace(angle_from, angle_to, 90))
    # Finding most prominent lines separated by a certain angle and distance
    _, theta_selected, ro_selected = hough_line_peaks(
        hspace_all, theta_all, rho_all, min_distance=9, min_angle=1, threshold=None, num_peaks=20
    )

    return np.array(theta_selected), np.array(ro_selected)


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


def draw_lines(image: np.ndarray, clusters: Optional[List[bool]]):
    """Draw lines on grayscale image on subplot's ax"""
    ymax, xmax = image.shape
    plt.figure(figsize=(8, 5))
    plt.imshow(image, cmap=cm.gray)

    def get_cmap(n, name="nipy_spectral"):
        """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name."""
        return plt.cm.get_cmap(name, n)

    def get_line(line):
        [(theta, rho)] = line
        # y = kx + b
        k, b = get_kb(theta, rho)
        x = np.linspace(0, xmax)
        y = k * x + b
        x, y = x[y < ymax], y[y < ymax]
        x, y = x[y > 0], y[y > 0]

        return x, y

    cmap = get_cmap(len(clusters))
    for c, lines in clusters.items():
        for line in lines[:-1]:
            x, y = get_line(line)
            color = cmap(c)
            plt.plot(x, y, color=color)
        x, y = get_line(lines[-1])
        color = cmap(c)
        plt.plot(x, y, color=color, label=f"{c}")
    plt.legend()
    plt.show()


def get_kb(theta: float, rho: float):
    # y = kx + b
    k = np.inf if np.tan(theta) == 0 else -1 / np.tan(theta)
    b = np.inf if np.sin(theta) == 0 else rho / np.sin(theta)

    return k, b


def get_homogeneous_coordinates(line: List[Tuple[float, float]]) -> List[float]:
    [(rho, theta)] = line
    return [np.round(np.cos(theta), 10), np.round(np.sin(theta), 10), -rho]


def get_intersection_point(line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]) -> List[float]:
    [a, b, c] = np.cross(get_homogeneous_coordinates(line1), get_homogeneous_coordinates(line2))

    return [a / c, b / c]


def get_intersection_line(line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]) -> List[float]:
    return np.cross(get_homogeneous_coordinates(line1), get_homogeneous_coordinates(line2))


def get_distance(point: Tuple[float, float], line: List[Tuple[float, float]]):
    [(rho, theta)] = line
    p_x, p_y = rho * np.cos(theta), rho * np.sin(theta)

    return abs(np.cos(theta) * (p_y - point[1]) - np.sin(theta) * (p_x - point[0]))


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
