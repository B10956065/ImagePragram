import cv2
import numpy as np
import matplotlib.pyplot as plt


def picture_basic(original_image, brightness, contrast, saturation):
    # https://stackoverflow.com/questions/50474302
    result = original_image * (contrast/127 + 1) - contrast + brightness
    result = np.clip(result, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    result[:, :, 1] = np.clip(result[:, :, 1] * saturation, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
    return result


def convex(src_image, effect) -> np.ndarray:
    """Realize fisheye effect with convex len effect.

    Args:
        src_image: the image will to be affect
        effect: parameter of the effect[x: int, y:int , radius: int]
    """
    row, col, channel = src_image.shape
    col = int(col)
    row = int(row)
    cx, cy, r = effect[:]
    output = np.zeros([row, col, channel], dtype=np.uint8)
    for y in range(row):
        for x in range(col):
            d = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) ** 0.5
            if d <= r:
                nx = int((x - cx) * d / r + cx)
                ny = int((y - cy) * d / r + cy)
                output[y, x, :] = src_image[ny, nx, :]
            else:
                output[y, x, :] = src_image[y, x, :]
    return output


def averageBlur(original_image, k_height, k_width):
    """

    Args:
        original_image:
        k_height:
        k_width:
    """
    result = cv2.blur(original_image, (k_height, k_width))
    return result


def medianBlur(original_image, k_size):
    if k_size < 1:
        print(f"k_size [{k_size} less than 1]")
        return original_image
    result = cv2.medianBlur(original_image, k_size)
    return result


def bilateralFilterBlur(original_image, sigma_color, sigma_space, d_radius: int = 5):
    result = cv2.bilateralFilter(original_image, d_radius, sigma_color, sigma_space)
    return result


def gaussianBlur(original_image, effect):
    """effect[kSize_height: int, kSize_width: int, sigmaX: int]"""
    result = cv2.GaussianBlur(original_image, (effect[0], effect[1]), effect[2])
    return result


def laplacian(original_image, depth, k_size, scale):
    result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    result = medianBlur(result, depth)
    result = cv2.Laplacian(result, -1, ksize=k_size, scale=scale)
    return result


def sobel(original_image, dx, dy, k_size, scale, depth=-1):  # FIXME: sobel k_size not work
    result = cv2.Sobel(original_image, depth, dx, dy, k_size, scale)
    return result


def canny(original_image, k_size, threshold1, threshold2):
    result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    result = cv2.Canny(gaussianBlur(result, [k_size, k_size, 0]), threshold1, threshold2)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def adaptiveThreshold(original_image, adaptiveMethod=1, thresholdType=cv2.THRESH_BINARY, maxValue=255, blockSize=11,
                      C=2, gBlur=1):
    result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    result = gaussianBlur(result, [gBlur, gBlur, 0])
    if adaptiveMethod == 0:
        adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    result = cv2.adaptiveThreshold(result, maxValue, adaptiveMethod, thresholdType, blockSize=blockSize, C=C)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def dilation(original_image, kSize_x, kSize_y, iterations=1):
    # result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kSize_x, kSize_y))
    # result = cv2.dilate(result, kernel, iterations=iterations)
    result = cv2.dilate(original_image, kernel, iterations=iterations)
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def erosion(original_image, kSize_x, kSize_y, iterations=1):
    # result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kSize_x, kSize_y))
    # result = cv2.erode(result, kernel, iterations=iterations)
    result = cv2.erode(original_image, kernel, iterations=iterations)
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def opening(original_image, kSize_x, kSize_y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kSize_x, kSize_y))
    # result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(original_image, cv2.MORPH_OPEN, kernel)
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def closing(original_image, kSize_x, kSize_y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kSize_x, kSize_y))
    # result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(original_image, cv2.MORPH_CLOSE, kernel)
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def topHat(original_image, kSize_x, kSize_y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kSize_x, kSize_y))
    # result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # result = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel)
    result = cv2.morphologyEx(original_image, cv2.MORPH_TOPHAT, kernel)
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def blackHat(original_image, kSize_x, kSize_y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kSize_x, kSize_y))
    # result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # result = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel)
    result = cv2.morphologyEx(original_image, cv2.MORPH_BLACKHAT, kernel)
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def morphologicalGradient(original_image, kSize_x, kSize_y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kSize_x, kSize_y))
    # result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
    result = cv2.morphologyEx(original_image, cv2.MORPH_GRADIENT, kernel)
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def saltAndPepperNoise(original_image, percent):
    percent /= 100
    mask = np.random.rand(*original_image.shape[:2])
    result = original_image.copy()
    result[mask < percent / 2] = 255
    result[mask > (1 - percent / 2)] = 0
    return result


def rotate(original_image, angle, center):
    result = original_image.copy()
    (height, width) = result.shape[:2]

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(result, M, (width, height))
    return result


def flipHorizontalVertical(original_image, pType):
    result = original_image.copy()
    if pType == 'flipHorizontal':
        pType = 1
    else:  # pType == 'flipVertical'
        pType = 0
    result = cv2.flip(result, pType)
    return result


# Improve underexposure or overexposure
def gammaCorrection(original_image, gamma):
    result = original_image.copy()
    result = np.power(result / 255.0, gamma) * 255.0
    result = np.uint8(result)
    return result


def negative(original_image):
    result = original_image.copy()
    result = 255 - result
    return result


def histogram(original_image):
    if original_image.ndim != 3:
        hist = cv2.calcHist([original_image], [0], None, [256], [0, 256])
        plt.plot(hist)
    else:
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist(original_image, [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
    plt.xlim([0, 256])
    plt.xlabel("Intensity")
    plt.ylabel("#Intensities")
    plt.show()
    return original_image


def histogramEqualization1(original_image):
    result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    result = cv2.equalizeHist(result)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def histogramEqualization2(oi, clipLimit=2, tileGridSize=(8, 8)):
    result = cv2.cvtColor(oi, cv2.COLOR_RGB2GRAY)
    clash = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    result = clash.apply(result)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def histogramEqualization3(oi, clipLimit=2, tileGridSize=(8, 8)):
    r, g, b = cv2.split(oi)
    clash = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    r = clash.apply(r)
    g = clash.apply(g)
    b = clash.apply(b)
    result = cv2.merge((r, g, b))
    return result
