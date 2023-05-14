import os
from typing import List, Tuple

import cv2


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
        for ImageName in os.listdir(input_image_path):
            # Reading images one by one.
            InputImage = cv2.imread(input_image_path + "/" + ImageName)

            images.append(InputImage)  # Storing images.
            image_names.append(ImageName)  # Storing image's names.

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
