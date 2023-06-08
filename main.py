from tkinter import *
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilename
import xml.etree.ElementTree as ET

from PIL import Image, ImageTk
from matplotlib.pyplot import imread
import numpy as np  # TODO: numpy should not be used much, streamlined it.
import cv2

import myImageProgramFunction as mpf

# global variables
DEFAULT_IMAGE_PATH = 'BottlenoseDolphins.png'
flag_program_type = 'medianBlur'
COUNT_SCALE = 4
list_scale_value = list()
list_scale_control = list()
list_scale_label_title = list()
list_scale_label_value = list()


def menuAdd(parent, content, type_=0):
    """
    Add menu item
    Args:
        parent: parent
        content: list of menu item, content==id==def name
        type_: 0=command(default) | 1=cascade
    """
    if type_ == 0:
        for i in content:
            if i == "separator":
                parent.add_separator()
            else:
                print(f"{i} => {t[i]}")
                parent.add_command(label=t[i], command=lambda x=i: callback_menu(x))
    else:  # type_ == 1
        for i in content:
            parent.add_cascade(menu=i[0], label=t[i[1]])


def update_image(newImage, frame="edited"):
    newImage = ImageTk.PhotoImage(Image.fromarray(newImage))
    if frame == "edited":
        imageFrame_edited.config(image=newImage)
        imageFrame_edited.image = newImage
    else:  # frame == "original"
        imageFrame_ori.config(image=newImage)
        imageFrame_ori.image = newImage


def image_program(llist):
    print(f"{flag_program_type} => {llist}")
    func = getattr(mpf, flag_program_type)
    result = func(image(), llist)
    update_image(result)
    return result


def callback_menu_file_save():
    filetypes = [('Image', '*.png')]
    filename = asksaveasfilename(filetypes=filetypes, defaultextension=".png")
    if filename:
        photo = callback_scale(None)
        Image.fromarray(photo).save(fp=filename)
    else:
        print("SaveError: User Cancel")


def callback_menu_file_load():
    opened_image = imread(askopenfilename())
    image.reload(opened_image)
    # TODO: auto resize if new image is too large


class theImage:
    def __init__(self):
        self.original = ((imread(DEFAULT_IMAGE_PATH)) * 255).astype(np.uint8)
        self.image = self.original.copy()

    def __call__(self, *args, **kwargs):
        return self.image

    def reload(self, newImage):
        """Update the original image and convent to uint8.

        Args:
            newImage: the new image will replace old original-image

        Warnings:
            User should not invoke this function manual.
        """
        self.original = (newImage * 255).astype(np.uint8)
        update_image(self.original, frame='original')
        self.image = self.original.copy()
        callback_scale()

    def resize(self, newX, newY, resizeFormOriginal=False):
        """Resize the image, this will not affect original-image

        Args:
            newX: new X coordinate
            newY: new Y coordinate
            resizeFormOriginal: whether resize form original-image. If Ture, the edited part will be reset.
        """
        if resizeFormOriginal:
            resizeTarget = self.original
        else:
            resizeTarget = self
        oldX, oldY = self.original.shape[:2]
        if oldX * oldY < newX*newY:
            self.image = cv2.resize(self.original, (newX, newY), interpolation=cv2.INTER_AREA)
        else:
            self.image = cv2.resize(self.original, (newX, newY), interpolation=cv2.INTER_CUBIC)
        update_image(self.image)
        # update information-label-3 :original-size & edit original-size
        list_label_information[3].config(text=f"oriSize:{oldX}*{oldY}\nediSize:{newX}*{newY}")

    def reset(self):
        """Reset edited-image to original-image"""
        self.image = self.original.copy()
        # update information-label-3 :original-size & edit original-size
        list_label_information[3].config(text=f"oriSize:{self.original.shape[0]}*{self.original.shape[1]}\n"
                                              f"ediSize:{self.image.shape[0]}*{self.image.shape[1]}")

    def cover(self):
        self.image = callback_scale()


class IntegerInputDialog(Toplevel):
    def __init__(self, parent, number: int, content=None, title: str = "Input Dialog"):
        """

        Args:
            parent: ==mainframe
            number: the number of entry to add
            content: label text for each entry
            title: title of dialog frame

        Examples:
             dialog = IntegerInputDialog(parent=mainframe, number=2, title='Resize', content=['newX', 'newY'])
        """
        super().__init__(parent)

        self.flag_submit = False  # used to check whether the user submit in correct way
        self.llist = list()  # used to record the input number(IntVar())
        self.title(title)

        self.number = number
        self.content = content

        # autofill the content list if the len of content is not enough to match the number
        if self.content is None:
            self.content = []
        for i in range(len(self.content), self.number):
            self.content.append(f"title_{i}")

        # add entry
        for i in range(self.number):
            ttk.Label(self, text=self.content[i]).grid(row=i, column=0)
            self.llist.append(IntVar())
            ttk.Entry(self, textvariable=self.llist[i]).grid(row=i, column=1)
            self.llist[i].set(0)  # set default value

        # add button
        ttk.Button(self, text="  OK  ", command=self.on_ok).grid(row=self.number + 1, column=0)
        ttk.Button(self, text="Cancel", command=self.on_cancel).grid(row=self.number + 1, column=1)

        # add padding
        for i in self.winfo_children():
            i.grid_configure(padx=5, pady=5)

        self.transient(parent)
        self.grab_set()

    # if the user correctly submit
    def on_ok(self):
        self.flag_submit = True  # avoid user destroy the dialog without the button to cause bug
        self.destroy()

    # if the user correctly cancel
    def on_cancel(self):
        self.llist = None
        self.destroy()


def callback_cover():
    image.cover()


def callback_resize():
    dialog = IntegerInputDialog(parent=mainframe, number=2, title='Resize',
                                content=['newX', 'newY'])
    root.wait_window(dialog)

    if dialog.llist is not None and dialog.flag_submit:
        print("Input Value:")
        for i in range(dialog.number):
            print(f"  {dialog.content[i]} => {dialog.llist[i].get()}")
        image.resize(dialog.llist[0].get(), dialog.llist[1].get())
    else:
        print("You don't input any value")


def callback_reset(resize=False):
    image.reset()
    if resize:
        callback_resize()
    callback_scale()


def callback_scale(_=None):
    llist = list()
    for i in range(COUNT_SCALE):
        llist.append(int(list_scale_control[i].get()))
        list_scale_label_value[i].config(text=int(list_scale_control[i].get()))
    # print(f"scale => {llist}")
    result = image_program(llist)
    return result


def callback_menu(program_type: str):
    def location(scale, axis):
        list_scale_label_title[scale].config(text=n[flag_program_type][f'l{scale}'])  # setting title
        list_scale_control[scale].configure(from_=1, to=image().shape[int(axis)], state="normal")  # setting limit

    def value(scale, from_=0, to=100):
        """Set starting value and end value of the scale.

        Args:
            scale: If scale is a list, then the format must be: [[scale1, from_1, to1], [scale2, from_2, to2], ...]. \
                   Else use "from_" and "to"
            from_: value of start
            to:  value of end
        """
        list_scale_label_title[scale].config(text=n[flag_program_type][f'l{scale}'])  # setting title
        list_scale_control[scale].configure(from_=from_, to=to, state="normal")  # setting limit

    def disable(scale):
        list_scale_label_title[scale].config(text='disable')
        list_scale_control[scale].configure(state="disable")

    def update():
        pass

    global flag_program_type
    flag_program_type = program_type
    print(f"program_type => {program_type}")
    list_label_information[1].config(text=f"{t['mainframe_i1_type']}:{t[flag_program_type]}")

    c = 0
    for i in f[program_type]:
        if i['type'] == 'v':
            value(c, i['f'], i['t'])
        elif i['type'] == 'l':
            location(c, i['a'])
        else:  # i['type'] == 'd'
            disable(c)
        c += 1
    callback_scale()


# load xml file and program
t = dict()  # text
n = dict()  # scale's name
for item in ET.parse("lang/cht.xml").getroot().findall('item'):
    t[item.find('id').text] = item.find('text').text
    if item.attrib != {}:
        n[item.find('id').text] = item.attrib
f = dict()  # function setting
for item in ET.parse("data.xml").getroot().findall('item'):
    f[item.find('id').text] = list()
    for sub_item in item:
        if sub_item.attrib != {}:
            f[item.find('id').text].append(sub_item.attrib)

# root
root = Tk()
root.title(t['mainframe_title'])
root.resizable(False, False)

# main frame
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky='N, W, E, S')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# add some label for show information
list_label_information = list()
for i in range(0, 4+1):
    list_label_information.append(ttk.Label(mainframe, text=t[f'mainframe_i{i+1}']))
    list_label_information[i].grid(row=0, column=i)
"""row_1 information label:
    0: None. (too narrow)
    1: program-type
    2: None. (too narrow)
    3: image-size
    4: ???
"""

# add some menu for select function
root.option_add('*tearOff', False)
menubar = Menu(root)
menu_file = Menu(menubar)
menu_edit = Menu(menubar)
menu_edge = Menu(menubar)
menu_filter = Menu(menubar)
menu_color = Menu(menubar)
menu_special_effect = Menu(menubar)
menuAdd(menubar, type_=1, content=[[menu_file, 'menu_file'], [menu_edit, 'menu_edit'],
                                   [menu_filter, 'menu_filter'], [menu_edge, 'menu_edge'],
                                   [menu_color, 'menu_color'], [menu_special_effect, 'menu_special_effect']])

ripple_effect = lambda: print("Ripple Effect")
fisheye_effect = lambda: print("Fisheye Effect")
twirl_effect = lambda: print("Twirl Effect")

# menu_file
menu_file.add_command(label=t['menu_file_open'], command=callback_menu_file_load)
menu_file.add_command(label=t['menu_file_save'], command=callback_menu_file_save)

# menu_edit
menu_edit.add_command(label=t['image_cover'], command=callback_cover)
menu_edit.add_command(label=t['image_resize'], command=callback_resize)
menu_edit.add_command(label=t['image_reset'], command=lambda: callback_reset(resize=False))
menu_edit.add_command(label=t['image_reset_resize'], command=lambda: callback_reset(resize=True))
menuAdd(menu_edit, content=['locationSelect'])

# menu_edge
menuAdd(menu_edge, content=['laplacian', 'sobel', 'canny', 'watershedAlgorithm', 'grabCutAlgorithm'])
menuAdd(menu_edge, content=['harrisCornerDetection', 'shiTomasiCornerDetection'])

# menu_filter
menuAdd(menu_filter, content=['averageBlur', 'medianBlur', 'bilateralFilterBlur', 'gaussianBlur'])
menuAdd(menu_filter, content=['adaptiveThreshold', 'globalThreshold'])

# menu_color
menuAdd(menu_color, content=['RGB_model', 'CMY_model', 'HSI_model', 'HSV_model', 'YCrCb_model',
                             'RGB_histogram_equalization', 'HSV_histogram_equalization'])

# menu_special_effect
menuAdd(menu_special_effect, content=['radial_pixelation', 'ripple_effect', 'fisheye_effect', 'twirl_effect',
                                      'fuzzy_effect', 'motion_blur', 'radial_blur',
                                      'edge_preserving_filter', 'detail_enhancement', 'pencil_sketch', 'stylization'])

# starting enable
root.config(menu=menubar)

# the scale for input value
for count in range(1, COUNT_SCALE + 1):
    # label of scale's title
    list_scale_label_title.append(ttk.Label(mainframe, text=f"Test_{count}"))
    list_scale_label_title[count - 1].grid(row=count, column=0)

    list_scale_value.append(IntVar())
    list_scale_control.append(ttk.Scale(mainframe, orient=HORIZONTAL, length=150, variable=list_scale_value[count - 1],
                                        command=callback_scale, from_=0, to=100))
    list_scale_control[count - 1].grid(row=count, column=1)

    # label of scale's value
    list_scale_label_value.append(ttk.Label(mainframe, text="0"))
    list_scale_label_value[count - 1].grid(row=count, column=2)

# image show
ttk.Label(mainframe, text=t['mainframe_imageOri']).grid(row=1, column=3)
ttk.Label(mainframe, text=t['mainframe_imageEdi']).grid(row=1, column=4)
imageFrame_ori = ttk.Label(mainframe)
imageFrame_ori.grid(row=2, column=3, rowspan=3)
imageFrame_edited = ttk.Label(mainframe)
imageFrame_edited.grid(row=2, column=4, rowspan=3)

# way to go
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

# init image frame
image = theImage()
image.resize(400, 266)
update_image(image(), frame="ori")

if __name__ == '__main__':
    root.mainloop()
