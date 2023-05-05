from tkinter import *
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilename
import xml.etree.ElementTree as ET

from PIL import Image, ImageTk
# import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, show
import numpy as np
import cv2

import myImageProgramFunction as mpf

# global variables
DEFAULT_IMAGE_PATH = 'BottlenoseDolphins.png'
flag_program_type = ''
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
    newImage = ImageTk.PhotoImage(Image.fromarray((newImage * 255).astype(np.uint8)))
    if frame == "edited":
        imageFrame_edited.config(image=newImage)
        imageFrame_edited.image = newImage
    else:
        imageFrame_ori.config(image=newImage)
        imageFrame_ori.image = newImage


def image_program(llist):
    print(f"{flag_program_type} => {llist}")
    func = getattr(mpf, flag_program_type)
    result = func(image(), llist)
    update_image(result)


class theImage:
    def __init__(self):
        self.original = (255-(imread(DEFAULT_IMAGE_PATH)*255)).astype(np.uint8)
        self.image = self.original.copy()

    def __call__(self, *args, **kwargs):
        return self.image

    def resize(self, newX, newY):
        # new = IntegerInputDialog(parent=mainframe, number=2, content=["new width", "new height"], title="resize")
        # if new.llist is not None and new.flag_submit:
        #     newX = new.llist[0].get()
        #     newY = new.llist[1].get()
        # else:  # incorrect submit
        #     print("Error: You didn't input any value")
        #     return -99
        self.image = cv2.resize(self.image, (newX, newY))
        update_image(self.image)


class IntegerInputDialog(Toplevel):
    def __init__(self, parent, number: int, content=None, title: str = "Input Dialog"):
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


def callback_scale(_):
    llist = list()
    for i in range(COUNT_SCALE):
        llist.append(int(list_scale_control[i].get()))
        list_scale_label_value[i].config(text=int(list_scale_control[i].get()))
    # print(f"scale => {llist}")
    image_program(llist)


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

    pass
    print(f"program_type => {program_type}")
    global flag_program_type
    flag_program_type = program_type
    c = 0
    for i in f[program_type]:
        if i['type'] == 'v':
            value(c, i['f'], i['t'])
        elif i['type'] == 'l':
            location(c, i['a'])
        else:  # i['type'] == 'd'
            disable(c)
        c += 1
    pass


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
ttk.Label(mainframe, text=t['mainframe_i1']).grid(row=0, column=0)
ttk.Label(mainframe, text=t['mainframe_i2']).grid(row=0, column=1)
ttk.Label(mainframe, text=t['mainframe_i3']).grid(row=0, column=2)
ttk.Label(mainframe, text=t['mainframe_i4']).grid(row=0, column=3)
ttk.Label(mainframe, text=t['mainframe_i5']).grid(row=0, column=4)

# add some menu for select function
root.option_add('*tearOff', False)
menubar = Menu(root)
menu_file = Menu(menubar)
menu_edit = Menu(menubar)
menu_edge = Menu(menubar)
menu_filter = Menu(menubar)
menuAdd(menubar, type_=1, content=[[menu_file, 'menu_file'], [menu_edit, 'menu_edit'],
                                   [menu_filter, 'menu_filter'], [menu_edge, 'menu_edge']])

# menu_file
pass

# menu_edit
pass

# menu_edge
pass

# menu_filter
menuAdd(menu_filter, content=['averageBlur', 'medianBlur', 'bilateralFilterBlur', 'gaussianBlur'])

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
