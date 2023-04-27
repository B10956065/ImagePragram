from tkinter import *
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilename
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt

__all__ = [cv2]

# global variables
DEFAULT_IMAGE_PATH = 'BottlenoseDolphins.png'
flag_program_type = 'fisheye'
flag_continuous_cover = False


def picture_basic(original_image, brightness, contrast, saturation):
    result = cv2.add(original_image, np.ones(original_image.shape, dtype='uint8') * brightness)
    result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)
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


def sobel(original_image, depth, dx, dy, k_size, scale):  # FIXME: sobel k_size not work
    result = cv2.Sobel(original_image, -1, dx, dy, k_size, scale)
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


def callback_menu_filter_rgbTunnel(ttype):
    rgb = image.image.copy()
    if ttype == 'rgbTunnel_red':
        rgb[:, :, 1] = 0
        rgb[:, :, 2] = 0
    elif ttype == 'rgbTunnel_green':
        rgb[:, :, 2] = 0
        rgb[:, :, 0] = 0
    else:  # ttype == 'rgbTunnel_blue':
        rgb[:, :, 0] = 0
        rgb[:, :, 1] = 0
    return rgb


def gray(original_image=None):  # TODO: finish this new function!
    if original_image is None:
        original_image = image.image
    result = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def callback_menu_filter_binarization(ttype, thresh=127, maxValue=255):
    result = image.image.copy()
    result = gray(result)
    match ttype:
        case 'bin_binary':
            _, result = cv2.threshold(result, thresh, maxValue, cv2.THRESH_BINARY)
        case 'bin_binary_inv':
            _, result = cv2.threshold(result, thresh, maxValue, cv2.THRESH_BINARY_INV)
        case 'bin_trunc':
            _, result = cv2.threshold(result, thresh, maxValue, cv2.THRESH_TRUNC)
        case 'bin_toZero':
            _, result = cv2.threshold(result, thresh, maxValue, cv2.THRESH_TOZERO)
        case 'bin_toZero_inv':
            _, result = cv2.threshold(result, thresh, maxValue, cv2.THRESH_TOZERO_INV)
    return result


def saltAndPepperNoise(original_image, percent):
    noise = np.zeros_like(original_image)
    percent /= 100
    mask = np.random.rand(*original_image.shape[:2])
    result = original_image.copy()
    result[mask < percent / 2] = 255
    result[mask > (1 - percent / 2)] = 0
    # result = cv2.addWeighted(original_image, 0, noise, 1, 0)
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


def affine_transform():  # TODO: ??? how and what it work ???
    result = image.image.copy()
    height, width = result.shape[:2]
    pts1 = np.float32([[160, 165], [240, 390], [270, 125]])
    pts2 = np.float32([[190, 140], [190, 375], [310, 140]])
    T = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(result, T, (width, height))
    return result


def image_program(llist):
    # Main image program
    def update_image(image_new):
        image_waiting = Image.fromarray(image_new)
        image_waiting = ImageTk.PhotoImage(image=image_waiting)
        label_image.configure(image=image_waiting)
        label_image.image = image_waiting
        return image_new

    match flag_program_type:
        case 'fisheye':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(convex(image.image, tuple(llist)))
        case 'gaussianBlur':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
                if i <= 2 and llist[i] % 2 == 0:
                    llist[i] += 1
            return update_image(gaussianBlur(image.image, llist))
        case 'averageBlur':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(averageBlur(image.image, llist[0], llist[1]))
        case 'medianBlur':
            if llist[0] % 2 == 0:
                llist[0] += 1
            return update_image(medianBlur(image.image, llist[0]))
        case 'bilateralFilterBlur':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(bilateralFilterBlur(image.image, llist[0], llist[1]))
        case 'laplacian':
            for i in range(len(llist)):
                if llist[i] % 2 == 0:
                    llist[i] += 1
            return update_image(laplacian(image.image, llist[0], llist[1], llist[2]))
        case 'sobel':
            if llist[0] <= 1:
                llist[0] = [1, 0]
            elif llist[0] == 2:
                llist[0] = [0, 1]
            elif llist[0] >= 3:
                llist[0] = [1, 1]

            if llist[1] % 2 == 0:
                llist[1] += 1
            if llist[2] % 2 == 0:
                llist[2] += 1
            return update_image(sobel(image.image, -1, llist[0][0], llist[0][1], llist[1], llist[2]))
        case 'canny':
            if llist[0] % 2 == 0:
                llist[0] += 1
            return update_image(canny(image.image, llist[0], llist[1], llist[2]))
        case 'adaptiveThreshold':
            if llist[0] % 2 == 0:
                llist[0] += 1
            if llist[1] <= 1:
                llist[1] = 3
            elif llist[1] % 2 == 0:
                llist[1] += 1
            return update_image(adaptiveThreshold(image.image, gBlur=llist[0], blockSize=llist[1], C=llist[2]))
        case 'dilation':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(dilation(image.image, llist[0], llist[1], llist[2]))
        case 'erosion':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(erosion(image.image, llist[0], llist[1], llist[2]))
        case 'opening':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(opening(image.image, llist[0], llist[1]))
        case 'closing':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(closing(image.image, llist[0], llist[1]))
        case 'topHat':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(topHat(image.image, llist[0], llist[1]))
        case 'blackHat':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(blackHat(image.image, llist[0], llist[1]))
        case 'morphologicalGradient':
            for i in range(len(llist)):
                if llist[i] < 1:
                    llist[i] = 1
            return update_image(morphologicalGradient(image.image, llist[0], llist[1]))
        case 'rgbTunnel_red' | 'rgbTunnel_green' | 'rgbTunnel_blue':
            return update_image(callback_menu_filter_rgbTunnel(flag_program_type))
        case 'bin_binary' | 'bin_binary_inv' | 'bin_trunc' | 'bin_toZero' | 'bin_toZero_inv':
            return update_image(callback_menu_filter_binarization(flag_program_type, llist[0], llist[1]))
        case 'basic':
            llist[0] = llist[0] - 100
            llist[1] /= 100
            if llist[2] >= 100:
                llist[2] = 1 + (llist[2] - 100) / 10
            else:  # < 100
                llist[2] /= 100
            return update_image(picture_basic(image.image, llist[0], llist[1], llist[2]))
        case 'gray':
            return update_image(gray(image.image))
        case 'saltAndPepperNoise':
            return update_image((saltAndPepperNoise(image.image, llist[0])))
        case 'rotate':
            llist[0] = -(llist[0] - 360)
            return update_image(rotate(image.image, llist[0], (llist[1], llist[2])))
        case 'flipHorizontal' | 'flipVertical':
            return update_image(flipHorizontalVertical(image.image, flag_program_type))
        case 'affineTransform':
            return update_image(affine_transform())
        case 'gammaCorrection':
            return update_image(gammaCorrection(image.image, llist[0]))
        case 'negative':
            return update_image(negative(image.image))
        case 'histogram':
            return update_image(histogram(image.image))
        case 'histogramEqualization':
            llist[0] /= 10
            return update_image(histogramEqualization3(image.image, llist[0], (llist[1], llist[1])))


def callback_scale(_=None):
    scale_1_value.set(int(scale_1_value.get()))
    scale_2_value.set(int(scale_2_value.get()))
    scale_3_value.set(int(scale_3_value.get()))
    llist = [scale_1_value.get(), scale_2_value.get(), scale_3_value.get()]
    print(llist)
    result = image_program(llist)
    return result


class ImageProgram:
    def __init__(self, image_path):
        self.original_image = (plt.imread(image_path) * 255).astype(np.uint8)
        self.image = self.original_image
        self.resize(400, 266)

    def resize(self, new_x: int, new_y: int):
        """Resize the image, original_image will not be affected.\n
        If new size less than original size, interpolation will be cv2.INTER_AREA,
        elif new size more the original size, interpolation will be cv2.INTER_CUBIC

        Args:
            new_x: new width
            new_y: new height"""
        old_y, old_x = self.original_image.shape[:2]
        # TODO: Should the original image used for resize be "image" or "original_image"?
        if old_x * old_y < new_x * new_y:
            self.image = cv2.resize(self.original_image, (new_x, new_y), interpolation=cv2.INTER_AREA)
        else:
            self.image = cv2.resize(self.original_image, (new_x, new_y), interpolation=cv2.INTER_CUBIC)

        if 'scale_1_value' in globals():
            callback_scale(None)

        # update information label
        global label_image_size_value
        label_image_size_value.set(f"Now size: {new_x}*{new_y}\n"
                                   f"Real size: {old_x}*{old_y:}")

    def reload(self, new_image):
        """Reload the original_image and reset image.\n
        The original original_image will be replaced with new new_image.
        and after this, the image will be reset to conform to new original.\n
        Usually used to open new image.

        Args:
            new_image: new image
        """
        self.original_image = (new_image * 255).astype(np.uint8)
        self.reset()
        callback_scale(None)

    def reset(self):
        """reset image to original_image, the original_image will not be affected"""
        self.image = self.original_image
        # update information label
        global label_image_size_value
        label_image_size_value.set(f"Now size: {self.image.shape[1]}*{self.image.shape[0]}\n"
                                   f"Real size: {self.image.shape[1]}*{self.image.shape[0]:}")


class IntegerInputDialog(Toplevel):
    def __init__(self, parent, number: int, content=None, title: str = "Input Dialog"):
        """
        Call a dialog window to let user input value(integer only).
        Args:
            parent: the new window will be attached to the parent, usually is "mainframe"
            number: how many entry need to generate
            content: the title and default value for each entry. If number of item in "title" is less than "number",
                    the default title will be "title_x", x means the serial number of the entry.
                    And the default entry value will be 0.
                    "title" is list in list. The structure must be [[title1, value1], [title2, value2], ...]
        """
        super().__init__(parent)

        self.flag_submit = False
        self.llist = list()
        self.title(title)

        self.number = number
        self.content = content

        if self.content is None:
            self.content = []

        for i in range(number if number <= len(self.content) else len(self.content)):
            '''if number <= len(title), it means we have enough item to use.
            elif number > len(title), it means not enough, the not enough part will be supplemented in next stage.'''
            ttk.Label(self, text=self.content[i][0]).grid(row=i, column=0)
            self.llist.append(IntVar())  # generate a variable and record
            ttk.Entry(self, textvariable=self.llist[i]).grid(row=i, column=1)
            self.llist[i].set(self.content[i][1])  # set default value

        if number > len(self.content):
            for i in range(len(self.content), number):
                ttk.Label(self, text=f"title_{i}").grid(row=i, column=0)
                self.llist.append(IntVar())  # generate a variable and record
                ttk.Entry(self, textvariable=self.llist[i]).grid(row=i, column=1)
                self.llist[i].set(0)  # set default value
                self.content.append([f"title_{i}", 0])  # record auto-generate content

        button_ok = ttk.Button(self, text="OK", command=self.on_ok)
        button_ok.grid(row=number + 1, column=0)

        button_cancel = ttk.Button(self, text="Cancel", command=self.on_cancel)
        button_cancel.grid(row=number + 1, column=1)

        for i in self.winfo_children():
            i.grid_configure(padx=5, pady=5)  # add some padding for all widgets

        self.transient(parent)  # bind child window and parent window, child will auto-close if parend being closed
        self.grab_set()  # force user to process child window first

    def on_ok(self):
        self.flag_submit = True
        self.destroy()

    def on_cancel(self):
        self.llist = None  # clear data to avoid mistake
        self.destroy()


def callback_menu_file_open():
    image_path = askopenfilename()
    opened_image = plt.imread(image_path)
    image.reload(opened_image)
    # TODO: if new image is too large, auto resize()


def callback_menu_file_save():
    files = [('Image', '*.png'),
             ('All Files', '*.*'),
             ('Python Files', '*.py'),
             ('Text Document', '*.txt')]
    filename = asksaveasfilename(filetypes=files, defaultextension=".png")
    if filename:
        if image.image is not None:
            image_waiting = callback_scale(None)
            Image.fromarray(image_waiting).save(fp=filename)
        else:
            print("the Image that wait for saving is not exist")
    else:
        print("ImageName is not exist")


def callback_menu_file_close():
    root.destroy()


def callback_menu_edit_resize():
    dialog = IntegerInputDialog(parent=mainframe, content=[["new x", 1], ["new_y", 1]], number=2, title="resize")
    root.wait_window(dialog)

    if dialog.llist is not None and dialog.flag_submit:
        print("Input Value:")
        for i in range(dialog.number):
            print(f"{dialog.content[i][0]} => {dialog.llist[i].get()}")
        image.resize(dialog.llist[0].get(), dialog.llist[1].get())
    else:
        print("You don't input any value.")


def callback_menu_edit_cover():
    image.image = callback_scale()
    callback_scale()


def callback_menu_edit_reset():
    image.reset()
    callback_scale()


def callback_menu_edit_resetAndResize():
    image.reset()
    callback_menu_edit_resize()
    callback_scale()


def callback_menu_filter(program_type: str):
    """Adjust thw type of filter that need to program
     and the min value and max value of the scale according to the program type.

    Args:
        program_type: The type to be programed

    See Also:
        notepad_function.txt
    """

    def location(scale, axis: int):
        """
        Adjust the min value and max value of the scale. Or just disable.\n
        Adjust the value to the image's height and width.
        Args:
            scale: which scale to adjust
            axis: 0==Y; 1==X
        """
        scale.configure(from_=0, to=image.image.shape[axis], state="normal")

    def level(scale, start: int = 1, value: int = 255):
        """
        Adjust the min and max value of the scale to custom value.
        Args:
            scale: which scale to adjust.
            start: Minimum value.
            value: Maximum value.
        """
        if type(scale) is list:
            for i in scale:
                i.configure(from_=start, to=value, state="normal")
        else:
            scale.configure(from_=start, to=value, state="normal")

    def disable(scale):
        """
        Disable the scale.
        Args:
            scale: which scale to disable. Can be signal scale or a list.
        """
        if type(scale) is list:
            for i in scale:
                i.configure(state="disable")
        else:
            scale.configure(state="disable")

    def update_scale_label(llist=None):
        """Update the scales' title label

        Args:
            llist: the title of each scale. [title_1, title_2, title_3]
        """
        if llist is None or type(llist) is not list:
            llist = []

        if len(llist) < 3:
            for i in range(len(llist), 3):
                llist.append(f"untitled value {i + 1}")

        label_scale_1_value.set(llist[0])
        label_scale_2_value.set(llist[1])
        label_scale_3_value.set(llist[2])

    def update_scale_value(llist=None):
        """Set the value of each scale.

        Args:
            llist: new value for scale. => list[scale_1_new_value, scale_2_new_value, scale_3_new_value]
        """
        if llist is None:
            llist = [0, 0, 0]
        elif len(llist) < 3:
            for i in range(len(llist), 3):
                llist.append(0)

        scale_1_value.set(llist[0])
        scale_2_value.set(llist[1])
        scale_3_value.set(llist[2])

    print(program_type)
    global flag_program_type, label_program_type_value
    flag_program_type = program_type  # save the incoming program_type to the global variable
    label_program_type_value.set("Program type : " + program_type)  # adjust the label to show program_type

    # each case must have 3 function, all three "scale_control" must be adjusted.
    match flag_program_type:
        case 'fisheye':
            location(scale_1_control, 1)
            location(scale_2_control, 0)
            level(scale_3_control)
            update_scale_label(["X", "Y", "R"])
        case 'gaussianBlur':
            level([scale_1_control, scale_2_control, scale_3_control])
            update_scale_label(["k_h", "k_w", "sigma"])
        case 'averageBlur':
            level(scale_1_control)
            level(scale_2_control)
            disable(scale_3_control)
            update_scale_label(["height", "width", "None"])
        case 'medianBlur':
            level(scale_1_control)
            disable(scale_2_control)
            disable(scale_3_control)
            update_scale_label(["kSize", "None", "None"])
        case 'bilateralFilterBlur':
            level([scale_1_control, scale_2_control, scale_3_control])
            update_scale_label(["d", "sigma color", "sigma space"])
        case 'laplacian':
            level(scale_1_control, 1, 10)
            level(scale_2_control, 1, 31)  # kSize
            level(scale_3_control, 1, 100)
            update_scale_label()
        case 'sobel':
            level(scale_1_control, 2, 4)
            level(scale_2_control, 1, 31)  # kSize
            level(scale_3_control, 1, 10)
            update_scale_label()
        case 'canny':
            level(scale_1_control, 1, 31)
            level([scale_2_control, scale_3_control], 0, 255)
            update_scale_label(["gBlur", "threshold1", "threshold2"])
        case 'adaptiveThreshold':
            level(scale_1_control, 0, 31)
            level(scale_2_control, 3, 31)
            level(scale_3_control, 1, 31)
            update_scale_label(["gBlur", "block Size", "C"])
        case 'dilation':
            level([scale_1_control, scale_2_control, scale_3_control], 1, 31)
            update_scale_label(["kSize_X", "kSize_Y", "iterations"])
        case 'erosion':
            level([scale_1_control, scale_2_control, scale_3_control], 1, 31)
            update_scale_label(["kSize_X", "kSize_Y", "iterations"])
        case 'opening':
            level([scale_1_control, scale_2_control], 1, 31)
            disable(scale_3_control)
            update_scale_label(["kSize_X", "kSize_Y", "iterations"])
        case 'closing':
            level([scale_1_control, scale_2_control], 1, 31)
            disable(scale_3_control)
            update_scale_label(["kSize_X", "kSize_Y", "iterations"])
        case 'topHat':
            level([scale_1_control, scale_2_control], 1, 31)
            disable(scale_3_control)
            update_scale_label(["kSize_X", "kSize_Y", "iterations"])
        case 'blackHat':
            level([scale_1_control, scale_2_control], 1, 31)
            disable(scale_3_control)
            update_scale_label(["kSize_X", "kSize_Y", "iterations"])
        case 'morphologicalGradient':
            level([scale_1_control, scale_2_control], 1, 31)
            disable(scale_3_control)
            update_scale_label(["kSize_X", "kSize_Y", "iterations"])
        case 'rgbTunnel_red' | 'rgbTunnel_green' | 'rgbTunnel_blue':
            disable([scale_1_control, scale_2_control, scale_3_control])
            update_scale_label(['disable', 'disable', 'disable'])
        case 'bin_binary' | 'bin_binary_inv' | 'bin_trunc' | 'bin_toZero' | 'bin_toZero_inv':
            level([scale_1_control, scale_2_control], 0, 255)
            disable(scale_3_control)
            update_scale_label(['thresh', 'maxValue', 'disable'])
        case 'basic':
            level([scale_1_control, scale_2_control, scale_3_control], 1, 200)
            update_scale_label(['brightness', 'contrast', 'saturation'])
            update_scale_value([100, 100, 100])
        case 'gray':
            disable([scale_1_control, scale_2_control, scale_3_control])
            update_scale_label(['disable', 'disable', 'disable'])
            update_scale_value([0, 0, 0])
        case 'saltAndPepperNoise':
            level(scale_1_control, 1, 100)
            disable([scale_2_control, scale_3_control])
            update_scale_label(['percent', 'disable', 'disable'])
            update_scale_value([1])
        case 'rotate':
            level(scale_1_control, 0, 720)
            location(scale_2_control, 1)
            location(scale_3_control, 0)
            update_scale_label(['angle', 'centerX', 'centerY'])
            update_scale_value([360, int(image.image.shape[1] // 2), int(image.image.shape[0] // 2)])
        case 'flipHorizontal' | 'flipVertical':
            disable([scale_1_control, scale_2_control, scale_3_control])
            update_scale_label(['disable', 'disable', 'disable'])
        case 'affineTransform':
            disable([scale_1_control, scale_2_control, scale_3_control])
            update_scale_label(['disable', 'disable', 'disable'])
        case 'gammaCorrection':
            level(scale_1_control, 0, 10)
            disable([scale_2_control, scale_3_control])
            update_scale_label(['gamma', 'disable', 'disable'])
            update_scale_value([1, 0, 0])
        case 'negative':
            disable([scale_1_control, scale_2_control, scale_3_control])
            update_scale_label(['disable', 'disable', 'disable'])
        case 'histogram':
            disable([scale_1_control, scale_2_control, scale_3_control])
            update_scale_label(['disable', 'disable', 'disable'])
        case 'histogramEqualization':
            level(scale_1_control, 1, 160)
            level(scale_2_control, 1, 201)
            disable(scale_3_control)
            update_scale_label(['clipLimit', 'tileGridSize', 'disable'])
            update_scale_value([2, 8])
        case _:
            print(f"update_scale_label error: Can't match: {flag_program_type}")

    callback_scale(None)  # update in the last, otherwise there will be some magic bug! "Sobel" I'm talking you!


# root
root = Tk()
root.title("guiTest_tkinter")
root.resizable(False, False)

# main frame
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky='N, W, E, S')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# variable of information label
label_program_type_value = StringVar(value="Program type : " + flag_program_type)
label_image_size_value = StringVar(value="")

# load and initialize the image
image = ImageProgram(DEFAULT_IMAGE_PATH)

# Create a menu
root.option_add('*tearOff', False)
menubar = Menu(root)
menu_file = Menu(menubar)
menu_edit = Menu(menubar)
menu_edge = Menu(menubar)
menu_filter = Menu(menubar)
menubar.add_cascade(menu=menu_file, label='File')
menubar.add_cascade(menu=menu_edit, label='Edit')
menubar.add_cascade(menu=menu_filter, label='Filter')
menubar.add_cascade(menu=menu_edge, label='Edge')

menu_file.add_command(label='Open', command=callback_menu_file_open)
menu_file.add_command(label='Save', command=callback_menu_file_save)
menu_file.add_command(label='Close', command=callback_menu_file_close)

menu_edit.add_command(label="resize", command=callback_menu_edit_resize)
menu_edit.add_command(label="cover", command=callback_menu_edit_cover)
menu_edit.add_command(label="reset", command=callback_menu_edit_reset)
menu_edit.add_command(label="resetAndResize", command=callback_menu_edit_resetAndResize)
menu_edit.add_separator()
menu_edit.add_command(label="rotate", command=lambda: callback_menu_filter('rotate'))
menu_edit.add_command(label="flipHorizontal", command=lambda: callback_menu_filter('flipHorizontal'))
menu_edit.add_command(label="flipVertical", command=lambda: callback_menu_filter('flipVertical'))
menu_edit.add_command(label="affineTransform", command=lambda: callback_menu_filter('affineTransform'))

menu_filter.add_command(label='basic', command=lambda: callback_menu_filter('basic'))
menu_filter.add_command(label='Salt-and-pepper noise', command=lambda: callback_menu_filter('saltAndPepperNoise'))
menu_filter.add_separator()
menu_filter.add_command(label='fisheye', command=lambda: callback_menu_filter('fisheye'))
menu_filter.add_command(label='averageBlur', command=lambda: callback_menu_filter('averageBlur'))
menu_filter.add_command(label='medianBlur', command=lambda: callback_menu_filter('medianBlur'))
menu_filter.add_command(label='bilateralFilterBlur', command=lambda: callback_menu_filter('bilateralFilterBlur'))
menu_filter.add_command(label='gaussianBlur', command=lambda: callback_menu_filter('gaussianBlur'))
menu_filter.add_separator()
menu_filter_tunnel = Menu(menu_filter)
menu_filter.add_cascade(menu=menu_filter_tunnel, label='tunnel')
menu_filter.add_command(label='gray', command=lambda: callback_menu_filter('gray'))
menu_filter_binarization = Menu(menu_filter)
menu_filter.add_cascade(menu=menu_filter_binarization, label='binarization')
menu_filter.add_separator()
menu_filter.add_command(label='gammaCorrection', command=lambda: callback_menu_filter('gammaCorrection'))
menu_filter.add_command(label='negative', command=lambda: callback_menu_filter('negative'))
menu_filter.add_command(label='histogram', command=lambda: callback_menu_filter('histogram'))
menu_filter.add_command(label='histogramEqualization', command=lambda: callback_menu_filter('histogramEqualization'))

menu_filter_tunnel.add_command(label='red', command=lambda: callback_menu_filter('rgbTunnel_red'))
menu_filter_tunnel.add_command(label='green', command=lambda: callback_menu_filter('rgbTunnel_green'))
menu_filter_tunnel.add_command(label='blue', command=lambda: callback_menu_filter('rgbTunnel_blue'))

menu_filter_binarization.add_command(label='binary', command=lambda: callback_menu_filter('bin_binary'))
menu_filter_binarization.add_command(label='binary_inv', command=lambda: callback_menu_filter('bin_binary_inv'))
menu_filter_binarization.add_command(label='trunc', command=lambda: callback_menu_filter('bin_trunc'))
menu_filter_binarization.add_command(label='toZero', command=lambda: callback_menu_filter('bin_toZero'))
menu_filter_binarization.add_command(label='toZero_inv', command=lambda: callback_menu_filter('bin_toZero_inv'))

menu_edge.add_command(label='adaptiveThreshold', command=lambda: callback_menu_filter('adaptiveThreshold'))
menu_edge.add_separator()
menu_edge.add_command(label='dilation', command=lambda: callback_menu_filter('dilation'))
menu_edge.add_command(label='erosion', command=lambda: callback_menu_filter('erosion'))
menu_edge.add_command(label='opening', command=lambda: callback_menu_filter('opening'))
menu_edge.add_command(label='closing', command=lambda: callback_menu_filter('closing'))
menu_edge.add_command(label='topHat', command=lambda: callback_menu_filter('topHat'))
menu_edge.add_command(label='blackHat', command=lambda: callback_menu_filter('blackHat'))
menu_edge.add_command(label='morphologicalGradient', command=lambda: callback_menu_filter('morphologicalGradient'))
menu_edge.add_separator()
menu_edge.add_command(label='laplacian', command=lambda: callback_menu_filter('laplacian'))
menu_edge.add_command(label='sobel', command=lambda: callback_menu_filter('sobel'))
menu_edge.add_command(label='canny', command=lambda: callback_menu_filter('canny'))

root.config(menu=menubar)

# first label row
ttk.Label(mainframe, text='Here is some information').grid(row=0, column=0)
ttk.Label(mainframe, textvariable=label_program_type_value).grid(row=0, column=1)
ttk.Label(mainframe, textvariable=label_image_size_value).grid(row=0, column=2)

# scale 1
label_scale_1_value = StringVar()
ttk.Label(mainframe, textvariable=label_scale_1_value).grid(row=1, column=0)
scale_1_value = IntVar()
scale_1_control = ttk.Scale(mainframe, orient=HORIZONTAL, length=200, variable=scale_1_value, command=callback_scale)
scale_1_control.grid(row=1, column=1)
ttk.Label(mainframe, text=0, textvariable=scale_1_value).grid(row=1, column=2)

# scale 2
label_scale_2_value = StringVar()
ttk.Label(mainframe, textvariable=label_scale_2_value).grid(row=2, column=0)
scale_2_value = IntVar()
scale_2_control = ttk.Scale(mainframe, orient=HORIZONTAL, length=200, variable=scale_2_value, command=callback_scale)
scale_2_control.grid(row=2, column=1)
ttk.Label(mainframe, text=0, textvariable=scale_2_value).grid(row=2, column=2)

# scale 3
label_scale_3_value = StringVar()
ttk.Label(mainframe, textvariable=label_scale_3_value).grid(row=3, column=0)
scale_3_value = IntVar()
scale_3_control = ttk.Scale(mainframe, orient=HORIZONTAL, length=200, variable=scale_3_value, command=callback_scale)
scale_3_control.grid(row=3, column=1)
ttk.Label(mainframe, text=0, textvariable=scale_3_value).grid(row=3, column=2)

# label of image
ttk.Label(mainframe, text="Image").grid(row=4, column=0)
image_pil = Image.fromarray(image.image)
image_tk = ImageTk.PhotoImage(image=image_pil)
label_image = ttk.Label(mainframe, image=image_tk)
label_image.grid(row=4, column=1, columnspan=2)

# way to go
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

callback_menu_filter("fisheye")

root.mainloop()
