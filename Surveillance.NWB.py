#_____________________HEADER FILES______________________

import tkinter
from tkinter import*
from tkinter import ttk
from tkinter import filedialog
from _cffi_backend import callback
from PIL import ImageTk, Image
import cv2
from cv2 import *
import numpy as np
import sys
import time
import argparse
import imutils
from pathlib import Path

#_____________________USER-DEFINED FUNCTIONS______________________

kernel_d = np.ones((3,3), np.uint8)
kernel_e = np.ones((3,3), np.uint8)
kernel_gauss = (3,3)
is_blur = True                          #initializing_boolean_variables
is_close = True                         #initializing_boolean_variables
is_draw_ct = False                      #initializing_boolean_variables
fac = 2                                 #initializing_integer_variables

#___________________INITALIZING THE GUI WINDOW______________________

   
window=Tk()
window.configure(background="grey64");
window.title("NWB")
window.resizable(0,0)
window.geometry('1300x680')

#Need to setup GUI to maximum screen size automatically and adjust widgets accordingly

#_______________SETTING VARIBALES TO CHECK STATE OF BUTTON (CHECKED OR UNCHECKED)______________________

clicked= StringVar()
chkValue1 = BooleanVar()
chkValue2 = BooleanVar()
current_value1 = IntVar()
current_value2 = IntVar()


def get_current_value1():
    return int('{}'.format(current_value1.get()))

def slider_changed1(event):
    value_label1.configure(text=get_current_value1())

slider_label1 = Label(window,text='CF1',font=("Times New Roman",12),fg="black",bg="grey64").place(x=842,y=52)
slider1 = ttk.Scale(window, from_=5,to=25, orient='horizontal', command=slider_changed1, variable=current_value1).place(x=890,y=50)
value_label1 = ttk.Label(window, text=get_current_value1())
value_label1.place(x=995,y=52)

#Need to pass value of slider dynamically to objdetect()

def get_current_value2():
    return int('{}'.format(current_value2.get()))

def slider_changed2(event2):
    value_label2.configure(text=get_current_value2())

slider_label2 = Label(window,text='CF2',font=("Times New Roman",12),fg="black",bg="grey64").place(x=842,y=82)
slider2 = ttk.Scale(window, from_=5,to=25, orient='horizontal', command=slider_changed2, variable=current_value2).place(x=890,y=82)
value_label2 = ttk.Label(window, text=get_current_value2())
value_label2.place(x=995,y=82)


#_____________________CREATING BUTTONS______________________

title = Label(window, text = "Network Without Borders",font=("Times New Roman",18, 'bold'),fg="black",bg="grey64").place(x=495, y=10)
label_file_explorer = Label(window, text = "", fg = "blue")
label_file_explorer.grid(column = 1, row = 1)

#____________________ADDING FUNCTIONALITES_________________________

def browseFiles():
   source_file = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes =[('All Files', '.*')],parent=window)
   label_file_explorer.configure(text=""+source_file)

    def drawRectangle(frame, minus_frame):
            if(is_blur):
                    minus_frame = GaussianBlur(minus_frame, kernel_gauss, 0)
            minus_Matrix = np.float32(minus_frame)	
            if(is_close):
                    for i in range(get_current_value1()):
                            minus_Matrix = dilate(minus_Matrix, kernel_d)
                    
                    for i in range(get_current_value2()):
                            minus_Matrix = erode(minus_Matrix, kernel_e)
                    
            minus_Matrix = np.clip(minus_Matrix, 0, 255)
            minus_Matrix = np.array(minus_Matrix, np.uint8)
            contours, hierarchy = findContours(minus_Matrix.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)
            for c in contours:
                    (x, y, w, h) = boundingRect(c)	
                    rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if( is_draw_ct ):
                            drawContours(frame, contours, -1, (0, 255, 255), 2)
            imshow('result', frame)
            

#Move objdetect() as global function and need to pass file path as returned from browseFiles() to objdetect()
#Need to fit the  imshow('result', frame) inside the GUI (instead of pop up)

    def objdetect():
            capture = VideoCapture(str(source_file));
            while(1):
                    (ret_old, old_frame) = capture.read()                    
                    gray_oldframe = cvtColor(old_frame, COLOR_BGR2GRAY)
                    if(is_blur):
                            gray_oldframe = GaussianBlur(gray_oldframe, kernel_gauss, 0)
                    oldBlurMatrix = np.float32(gray_oldframe)
                    accumulateWeighted(gray_oldframe, oldBlurMatrix, 0.003)
                    while(True):
                            ret, frame = capture.read()                            
                            gray_frame = cvtColor(frame, COLOR_BGR2GRAY)
                            if(is_blur):
                                    newBlur_frame = GaussianBlur(gray_frame, kernel_gauss, 0)
                            else:
                                    newBlur_frame = gray_frame
                            newBlurMatrix = np.float32(newBlur_frame)
                            minusMatrix = absdiff(newBlurMatrix, oldBlurMatrix)
                            ret, minus_frame = threshold(minusMatrix, 60, 255.0, THRESH_BINARY)
                            accumulateWeighted(newBlurMatrix,oldBlurMatrix,0.02)
                            imshow('Input', frame)
                            drawRectangle(frame, minus_frame)
                            if cv2.waitKey(60) & 0xFF == ord('q'):
                                    break
                    capture.release() 
                    cv2.destroyAllWindows()

    objdetect()
   
   
C1=Button(window,text = "Browse",font=("Times New Roman",12, 'bold'),command=browseFiles).place(x=100,y=10)
C2=Button(window,text="Live Input",font=("Times New Roman",12, 'bold'),state=DISABLED).place(x=300,y=10)
C3=Button(window,text = "Object Detection",font=("Times New Roman",12, 'bold')).place(x=880,y=10)
C4=Button(window,text="Turbulence Mitigation",font=("Times New Roman",12, 'bold')).place(x=1090,y=10)


#___________________FOOTER OF THE GUI WINDOW______________________

frame=LabelFrame(window,width=1300, height=50,fg="black",bg="aqua").place(x=0,y=630)
foot=Label(frame,text = "Surveillance System",font=("Times New Roman",11),fg="black",bg="aqua").place(x=1010,y=645)
window.mainloop()

#____________________END OF PROGRAM______________________
