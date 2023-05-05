import cv2
import numpy as np

"""llist:
list for all parameter, whether it's used or not. Should not greater than 4, allow => [0, 1, 2, 3 ].
"""


def averageBlur(original, llist):
    """Average Blue, fast blur but lost detail

    Args:
        original: the original image you want to edit
        llist: list for all parameter
    """

    result = cv2.blur(original, (llist[0], llist[1]))
    return result


def medianBlur(original, llist):
    if llist[0] % 2 == 0:
        llist[0] += 1
    result = cv2.medianBlur(original, llist[0])
    return result


def bilateralFilterBlur(original_image, llist):
    result = cv2.bilateralFilter(original_image, d=5, sigmaColor=llist[0], sigmaSpace=llist[1])
    return result


def gaussianBlur(original_image, llist):
    # TODO: Numerical verification
    result = cv2.GaussianBlur(original_image, ksize=(llist[0], llist[1]), sigmaX=llist[2])
    return result
