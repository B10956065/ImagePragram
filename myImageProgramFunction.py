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
    temp = cv2.Laplacian(original, cv2.CV_32F) + 128
    result = np.uint8(np.clip(temp, 0, 255))
    return result


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


def harrisCornerDetection(original, llist):
    # llist[0]: block_size: (1~32)
    if llist[1] % 2 == 0:  # llist[1]: kSize: (1~31) odd, less than 31
        llist[1] += 1
    llist[2] /= 100  # llist[2]: k (1~50) => (0.01~0.5)
    llist[3] /= 100  # llist[3]: threshold (1~100) => (0.01~1)
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    dst = cv2.cornerHarris(gray, llist[0], llist[1], llist[2])
    dst = cv2.dilate(dst, None)

    result = original.copy()
    result[dst > llist[3] * dst.max()] = [255, 0, 0]
    return result


def shiTomasiCornerDetection(original, llist):
    llist[1] /= 100  # quality_level: (1~100) => (0.01~1)
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, llist[0], llist[1], llist[2])
    corners = np.int0(corners)

    result = original.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
    return result


def keypointDetection_SIFT(original, llist):
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoint = sift.detect(gray, None)
    result = cv2.drawKeypoints(original, keypoint, None,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return result


def keypointDetection_SURF(original, llist):
    print("SURF model is under patent protection, will be replaced by SIFT")
    result = keypointDetection_SIFT(original, llist)
    return result


def keypointDetection_ORB(original, llist):
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    keypoint = orb.detect(gray, None)
    keypoint = sorted(keypoint, key=lambda x: -x.response)[:llist[0]]  # 根據response進行排序
    result = cv2.drawKeypoints(original, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return result


def gray_level(original, llist):
    result = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    return result


def RGB_model(original, llist):
    channel = llist[0]
    # Red
    if channel == 1:
        return original[:, :, 0]
    # Green
    elif channel == 2:
        return original[:, :, 1]
    # Blue
    elif channel == 3:
        return original[:, :, 2]


def CMY_model(original, llist):
    channel = llist[0]
    # Cyan
    if channel == 1:
        return 255 - original[:, :, 0]
    # Magenta
    elif channel == 2:
        return 255 - original[:, :, 1]
    # Yellow
    elif channel == 3:
        return 255 - original[:, :, 2]


def HSV_model(original, llist):
    channel = llist[0]
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    # Hue
    if channel == 1:
        return hsv[:, :, 0]
    # Saturation
    elif channel == 2:
        return hsv[:, :, 1]
    # Value
    elif channel == 3:
        return hsv[:, :, 2]


def YCrCb_model(original, llist):
    channel = llist[0]
    ycrcb = cv2.cvtColor(original, cv2.COLOR_RGB2YCrCb)
    # Y
    if channel == 1:
        return ycrcb[:, :, 0]
    # Cr
    elif channel == 2:
        return ycrcb[:, :, 1]
    # Cb
    elif channel == 3:
        return ycrcb[:, :, 2]


def RGB_histogram_equalization(original, llist):
    result = original.copy()
    for k in range(3):
        result[:, :, k] = cv2.equalizeHist(original[:, :, k])
    return result


def HSV_histogram_equalization(original, llist):
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return result


def autumn(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_AUTUMN)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def bone(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_BONE)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def jet(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_JET)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def winter(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_WINTER)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def rainbow(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_RAINBOW)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def ocean(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_OCEAN)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def summer(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_SUMMER)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def spring(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_SPRING)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def cool(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_COOL)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def hsv(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_HSV)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def pink(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_PINK)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def hot(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_HOT)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def parula(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_PARULA)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def magma(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_MAGMA)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def inferno(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_INFERNO)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def plasma(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_PLASMA)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def viridis(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_VIRIDIS)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def cividis(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_CIVIDIS)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def twilight(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_TWILIGHT)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def twilight_shifted(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_TWILIGHT_SHIFTED)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def turbo(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_TURBO)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def deepgreen(original, llist):
    colormap = cv2.applyColorMap(original, cv2.COLORMAP_DEEPGREEN)
    result = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    return result


def ripple_effect(original, llist):
    method = llist[0]
    amplitude = llist[1]
    period = llist[2]

    nr, nc = original.shape[:2]
    map_x = np.zeros([nr, nc], dtype='float32')
    map_y = np.zeros([nr, nc], dtype='float32')
    x0, y0 = nr // 2, nc // 2
    for x in range(nr):
        for y in range(nc):
            # x-direction
            if method == 1:
                xx = np.clip(x + amplitude * np.sin(x / period), 0, nr - 1)
                map_x[x, y] = y
                map_y[x, y] = xx
            # y-direction
            elif method == 2:
                yy = np.clip(y + amplitude * np.sin(y / period), 0, nc - 1)
                map_x[x, y] = yy
                map_y[x, y] = x
            # x & y direction
            elif method == 3:
                xx = np.clip(x + amplitude * np.sin(x / period), 0, nr - 1)
                yy = np.clip(y + amplitude * np.sin(y / period), 0, nc - 1)
                map_x[x, y] = yy
                map_y[x, y] = xx
            # Radial
            elif method == 4:
                r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                if r == 0:
                    theta = 0
                else:
                    theta = np.arccos((x - x0) / r)
                r = r + amplitude * np.sin(r / period)
                if y - y0 < 0:
                    theta = -theta
                map_x[x, y] = np.clip(y0 + r * np.sin(theta), 0, nc - 1)
                map_y[x, y] = np.clip(x0 + r * np.cos(theta), 0, nr - 1)
    result = cv2.remap(original, map_x, map_y, cv2.INTER_LINEAR)
    return result


def fisheye_effect(original, llist):
    row, col, channel = original.shape
    col = int(col)
    row = int(row)
    cx, cy, r = llist[0:2+1]
    result = np.zeros([row, col, channel], dtype=np.uint8)
    for y in range(row):
        for x in range(col):
            d = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) ** 0.5
            if d <= r:
                nx = int((x - cx) * d / r + cx)
                ny = int((y - cy) * d / r + cy)
                result[y, x, :] = original[ny, nx, :]
            else:
                result[y, x, :] = original[y, x, :]
    return result


def radial_pixelation(original, llist):
    delta_r = llist[0]
    delta_theta = llist[1]
    nr, nc = original.shape[:2]
    map_x = np.zeros([nr, nc], dtype='float32')
    map_y = np.zeros([nr, nc], dtype='float32')
    x0, y0 = nr // 2, nc // 2
    for x in range(nr):
        for y in range(nc):
            r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if r == 0:
                theta = 0
            else:
                theta = np.arccos((x - x0) / r)
            r = r - r % delta_r
            if y - y0 < 0:
                theta = -theta
            theta = theta - theta % (np.radians(delta_theta))
            map_x[x, y] = np.clip(y0 + r * np.sin(theta), 0, nc - 1)
            map_y[x, y] = np.clip(x0 + r * np.cos(theta), 0, nr - 1)
    result = cv2.remap(original, map_x, map_y, cv2.INTER_LINEAR)
    return result


def twirl_effect(original, llist):
    k = llist[0]
    nr, nc = original.shape[:2]
    map_x = np.zeros([nr, nc], dtype='float32')
    map_y = np.zeros([nr, nc], dtype='float32')
    x0, y0 = nr // 2, nc // 2
    for x in range(nr):
        for y in range(nc):
            r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if r == 0:
                theta = 0
            else:
                theta = np.arccos((x - x0) / r)
            if y - y0 < 0:
                theta = -theta
            phi = theta + r / k
            map_x[x, y] = np.clip(y0 + r * np.sin(phi), 0, nc - 1)
            map_y[x, y] = np.clip(x0 + r * np.cos(phi), 0, nr - 1)
    result = cv2.remap(original, map_x, map_y, cv2.INTER_LINEAR)
    return result


def fuzzy_effect(original, llist):
    from numpy.random import uniform
    windowSize = llist[0]
    result = original.copy()
    nr, nc = original.shape[:2]
    for x in range(nr):
        for y in range(nc):
            xp = int(x + windowSize * uniform() - windowSize // 2)
            yp = int(y + windowSize * uniform() - windowSize // 2)
            xp = np.clip(xp, 0, nr - 1)
            yp = np.clip(yp, 0, nc - 1)
            result[x, y] = original[xp, yp]
    return result


def motion_blur(original, llist):
    length = llist[0]
    angle = llist[1]
    filter = np.zeros([length, length])
    x0, y0 = length // 2, length // 2
    x_len = round(x0 * np.cos(np.radians(angle)))
    y_len = round(y0 * np.sin(np.radians(angle)))
    x1, y1 = int(x0 - x_len), int(y0 - y_len)
    x2, y2 = int(x0 + x_len), int(y0 + y_len)
    cv2.line(filter, (y1, x1), (y2, x2), (1, 1, 1))
    filter /= np.sum(filter)
    result = cv2.filter2D(original, -1, filter)
    return result


def radial_blur(original, llist):
    filter_size = llist[0]
    result = original.copy()
    nr, nc = original.shape[:2]
    x0, y0 = nr // 2, nc // 2
    half = filter_size // 2
    for x in range(nr):
        for y in range(nc):
            r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if r == 0:
                theta = 0
            else:
                theta = np.arccos((x - x0) / r)
            if y - y0 < 0:
                theta = -theta
            R = G = B = n = 0
            for k in range(-half, half + 1):
                phi = theta + np.radians(k)
                xp = int(round(x0 + r * np.cos(phi)))
                yp = int(round(y0 + r * np.sin(phi)))
                if 0 <= xp < nr and 0 <= yp < nc:
                    R += original[xp, yp, 2]
                    G += original[xp, yp, 1]
                    B += original[xp, yp, 0]
                    n += 1
            R = round(R / n)
            G = round(G / n)
            B = round(B / n)
            result[x, y, 2] = np.uint8(R)
            result[x, y, 1] = np.uint8(G)
            result[x, y, 0] = np.uint8(B)
    return result


def edge_preserving_filter(original, llist):
    flags = llist[0]
    sigma_s = llist[1]
    sigma_r = llist[2] / 100
    result = cv2.edgePreservingFilter(original, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
    return result


def detail_enhancement(original, llist):
    sigma_s = llist[0]
    sigma_r = llist[1] / 100
    result = cv2.detailEnhance(original, sigma_s=sigma_s, sigma_r=sigma_r)
    return result


def pencil_sketch(original, llist):
    sigma_s = llist[0]
    sigma_r = llist[1] / 100
    shade_factor = llist[2] / 100
    result_pencil, result_color = cv2.pencilSketch(original, sigma_s=sigma_s, sigma_r=sigma_r,
                                                   shade_factor=shade_factor)
    return result_pencil


def stylization(original, llist):
    sigma_r = llist[0]
    sigma_s = llist[1] / 100
    result = cv2.stylization(original, sigma_r=sigma_r, sigma_s=sigma_s)
    return result


def uniform_noise(original, llist):
    from numpy.random import uniform
    scale = llist[0]
    result = original.copy()
    nr, nc = original.shape[:2]
    for x in range(nr):
        for y in range(nc):
            value = original[x, y] + uniform(0, 1) * scale
            result[x, y] = np.uint8(np.clip(value, 0, 255))
    return result


def gaussian_noise(original, llist):
    from numpy.random import uniform
    scale = llist[0]
    result = original.copy()
    nr, nc = original.shape[:2]
    for x in range(nr):
        for y in range(nc):
            value = original[x, y] + uniform(0, scale)
            result[x, y] = np.uint8(np.clip(value, 0, 255))
    return result


def exponential_noise(original, llist):
    from numpy.random import exponential
    scale = llist[0]
    result = original.copy()
    nr, nc = original.shape[:2]
    for x in range(nr):
        for y in range(nc):
            value = original[x, y] + exponential(scale)
            result[x, y] = np.uint8(np.clip(value, 0, 255))
    return result


def rayleigh_noise(original, llist):
    from numpy.random import rayleigh
    scale = llist[0]
    result = original.copy()
    nr, nc = original.shape[:2]
    for x in range(nr):
        for y in range(nc):
            value = original[x, y] + rayleigh(scale)
            result[x, y] = np.uint8(np.clip(value, 0, 255))
    return result


def salt_pepper_noise(original, llist):
    from numpy.random import uniform
    probability = llist[0] / 100
    result = original.copy()
    nr, nc = original.shape[:2]
    for x in range(nr):
        for y in range(nc):
            value = uniform(0, 1)
            if 0 < value <= probability / 2:
                result[x, y] = 0
            elif probability / 2 < value <= probability:
                result[x, y] = 255
            else:
                result[x, y] = original[x, y]
    return result


def negative(original, llist):
    result = 255 - original
    return result


def gamma_correction(original, llist):
    gamma = llist[0] / 100
    result = original.copy()
    nr, nc = original.shape[:2]
    c = 255.0 / (255.0 ** gamma)
    table = np.zeros(256)

    for i in range(256):
        table[i] = round(i ** gamma * c, 0)

    if original.ndim != 3:
        for x in range(nr):
            for y in range(nc):
                result[x, y] = table[original[x, y]]
    else:
        for x in range(nr):
            for y in range(nc):
                for k in range(3):
                    result[x, y, k] = table[original[x, y, k]]

    return result


def beta_correction(original, llist):
    import scipy.special as special
    a = llist[0] / 100
    b = llist[1] / 100
    result = original.copy()
    nr, nc = original.shape[:2]
    x = np.linspace(0, 1, 256)

    table = np.round(special.betainc(a, b, x) * 255, 0)

    if original.ndim != 3:
        for x in range(nr):
            for y in range(nc):
                result[x, y] = table[original[x, y]]
    else:
        for x in range(nr):
            for y in range(nc):
                for k in range(3):
                    result[x, y, k] = table[original[x, y, k]]

    return result
