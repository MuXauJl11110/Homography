import os
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian

from line import get_homogeneous_coordinates, get_intersection_point


def read_image(input_image_path: str) -> Tuple[List[cv2.Mat], List[str]]:
    images = []  # Input Images will be stored in this list.
    image_names = []  # Names of input images will be stored in this list.

    # Checking if path is of file or folder.
    if os.path.isfile(input_image_path):  # If path is of file.
        input_image = cv2.imread(input_image_path)  # Reading the image.

        # Checking if image is read.
        if input_image is None:
            print("Image not read. Provide a correct path")
            exit()

        images.append(input_image)  # Storing the image.
        image_names.append(os.path.basename(input_image_path))  # Storing the image's name.

    # If path is of a folder containing images.
    elif os.path.isdir(input_image_path):
        # Getting all image's name present inside the folder.
        for image_name in os.listdir(input_image_path):
            # Reading images one by one.
            input_image = cv2.imread(input_image_path + "/" + image_name)

            images.append(input_image)  # Storing images.
            image_names.append(image_name)  # Storing image's names.

    # If it is neither file nor folder(Invalid Path).
    else:
        print("\nEnter valid Image Path.\n")
        exit()

    return images, image_names


# https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
def show_image(window_name: str, image: cv2.Mat):
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # close window when a key press is detected
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)


def preprocess_image(image_name: str, plot: Optional[bool] = False) -> np.ndarray:
    image = io.imread(image_name)
    # Converting to grayscale
    gray_image = rgb2gray(image)
    # Blurring image to reduce noise.
    blur_gray_image = gaussian(gray_image, sigma=1, truncate=1 / 5)
    # Generating Edge image
    edge_image = canny(blur_gray_image, sigma=2)
    if plot:
        _, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax = ax.flatten()
        ax[0].imshow(image)
        ax[0].set_title("Original image")
        ax[1].imshow(gray_image, cmap=cm.gray)
        ax[1].set_title("Grayscale image")
        ax[2].imshow(blur_gray_image, cmap=cm.gray)
        ax[2].set_title("Blurred image")
        ax[3].imshow(edge_image, cmap=cm.gray)
        ax[3].set_title("Edge image")
        plt.show()

    return edge_image, gray_image


def show_image_with_lines(image: cv2.Mat, thetas: np.ndarray, rhos: np.ndarray, inliers: Optional[List[bool]] = None):
    ymax, _, _ = image.shape
    homogeneous_lines = [get_homogeneous_coordinates(theta, rho) for (theta, rho) in zip(thetas, rhos)]

    if inliers is None:
        inliers = [True] * len(thetas)
    for line, inl in zip(homogeneous_lines, inliers):
        x1, y1 = get_intersection_point(line, [0, 1, 0])
        x2, y2 = get_intersection_point(line, [0, 1, -ymax])
        color = "green" if inl else "red"
        # plt.plot(x1, y1, x2, y2, color=color linewidth=3)

    plt.imshow(data, cmap="grey")
    plt.show()
