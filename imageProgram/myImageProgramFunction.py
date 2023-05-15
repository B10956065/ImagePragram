import cv2
import numpy as np

"""Here have almost all the imageProgram-function used.

Args:
    original: original-image will used to program, but should not effect this or it will cause some error.
    llist: List for all parameter, whether it's used or not. Should not greater than 4, allow => [0, 1, 2, 3 ].
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
    """median blur"""
    if llist[0] % 2 == 0:
        llist[0] += 1
    result = cv2.medianBlur(original, llist[0])
    return result


def bilateralFilterBlur(original_image, llist):
    result = cv2.bilateralFilter(original_image, d=5, sigmaColor=llist[0], sigmaSpace=llist[1])
    return result


def gaussianBlur(original_image, llist):
    if llist[0] % 2 == 0:
        llist[0] += 1
    if llist[1] % 2 == 0:
        llist[1] += 1
    result = cv2.GaussianBlur(original_image, ksize=(llist[0], llist[1]), sigmaX=llist[2])
    return result


def laplacian(original, llist):
    # TODO: laplacian
    return original


def sobel(original, llist):
    depth = -1
    dx = 1
    dy = 1
    k_size = llist[0]
    scale = llist[1]
    if k_size % 2 == 0:
        k_size += 1
    if scale % 2 == 0:
        scale += 1
    result = cv2.Sobel(original, depth, dx, dy, k_size, scale)
    return result


def canny(original, llist):
    k_size = llist[0]
    threshold1 = llist[1]
    threshold2 = llist[2]
    if k_size % 2 == 0:
        k_size += 1
    result = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    result = cv2.Canny(gaussianBlur(result, [k_size, k_size, 0]), threshold1, threshold2)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def adaptiveThreshold(original, llist):
    gBlur = llist[0]
    blockSize = llist[1]
    C = llist[2]

    if blockSize <= 1:
        blockSize = 3
    elif blockSize % 2 == 0:
        blockSize += 1

    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType = cv2.THRESH_BINARY
    maxValue = 255
    result = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    result = gaussianBlur(result, [gBlur, gBlur, 0])
    result = cv2.adaptiveThreshold(result, maxValue, adaptiveMethod, thresholdType, blockSize=blockSize, C=C)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def globalThreshold(original, llist):
    thresholdValue = llist[0]
    maxValue = llist[1]
    _, result = cv2.threshold(original, thresholdValue, maxValue, cv2.THRESH_BINARY)
    return result


def watershedAlgorithm(original, llist):
    pValue = llist[0] / 10
    result = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    gradient = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel=np.ones((3, 3), np.uint8))
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, pValue * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers += 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(original, markers)
    result = original.copy()
    result[markers == -1] = [0, 0, 255]

    return result


def grabCutAlgorithm(original, llist):
    mask = np.zeros(original.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (llist[0], llist[1], llist[2], llist[3])

    cv2.grabCut(original, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    segmented = original * mask2[:, :, np.newaxis]
    cv2.rectangle(segmented, (llist[0], llist[1]), (llist[2], llist[3]), (255, 0, 0), 1)
    return segmented


def locationSelect(original, llist):
    for i in range(4):
        if llist[i] <= 0:
            llist[i] = i
    result = original.copy()
    result = cv2.rectangle(result, (llist[0], llist[1]), (llist[2], llist[3]), (255, 0, 0), 1)
    return result
