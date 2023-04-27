from tkinter import *
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilename

# from PIL import Image, ImageTk
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2

# global variables
DEFAULT_IMAGE_PATH = 'BottlenoseDolphins.png'
COUNT_SCALE = 4
list_scale_value = list()
list_scale_control = list()
list_scale_label = list()


def callback_scale(_):
    for i in range(COUNT_SCALE):
        print(f"scale_{i} => ", list_scale_control[i].get())
    print("-=-=-=-=-=-=-=-=-=-=-")


root = Tk()
root.title("Image Program")
root.resizable(False, False)
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky='N, W, E, S')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

for count in range(1, COUNT_SCALE + 1):
    list_scale_value.append(IntVar())
    list_scale_control.append(ttk.Scale(mainframe, orient=HORIZONTAL, length=200, variable=list_scale_value[count-1],
                                        command=callback_scale, from_=0, to=0))
    list_scale_control[count-1].grid(row=count, column=1)

# way to go
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()

