# _____________________HEADER FILES______________________

import tkinter
from tkinter import*
from tkinter import ttk
from tkinter import filedialog
from imutils.video import FPS
from _cffi_backend import callback
from PIL import ImageTk, Image
import cv2
from cv2 import *
import numpy as np
import os
import sys
import time
import argparse
import imutils
import keyboard
from pathlib import Path
from utils import *
import time
from skimage.restoration import wiener, richardson_lucy
from scipy.special import j1


# _____________________USER-DEFINED FUNCTIONS______________________

kernel_d = np.ones((3, 3), np.uint8)
kernel_e = np.ones((3, 3), np.uint8)
kernel_gauss = (3, 3)
is_blur = False  # initializing_boolean_variables
is_close = True  # initializing_boolean_variables
is_draw_ct = False  # initializing_boolean_variables
fac = 2  # initializing_integer_variables
isVideoCaptureOpen = False  # boolean flag to keep a check of the video capture

#______________________OUTPUT_________________________________________

parent_dir = os.getcwd() 
directory = "results"
path = os.path.join(parent_dir,directory)
try:
    os.mkdir(path)
except OSError as error:
    pass    

# ___________________INITALIZING THE GUI WINDOW______________________

window = Tk()
window.configure(background="grey64")
window.title("Surveillance System")
window.resizable(height=None, width=None)
window.geometry('1300x480')

# _______________SETTING VARIBALES TO CHECK STATE OF BUTTON (CHECKED OR UNCHECKED)______________________


current_value1 = IntVar()
current_value2 = IntVar()

# _______________________Global Variables__________________________

capture = VideoCapture(0)
frame_width = 0
frame_height = 0
ROI_enhanced_arr = []
Combined_frames = []
object_frames = []
iterFPS =0
fps = 0
fat = 0


def get_current_value1():
    # print("Dilation value 1: ", current_value1.get())
    return int('{}'.format(current_value1.get()))


def slider_changed1(event):
    value_label1.configure(text=get_current_value1())


slider_label1 = Label(window, text='Dilation', font=(
    "Times New Roman", 12), fg="black", bg="grey64").place(x=832, y=52)
value_label1 = ttk.Label(window, text=get_current_value1())
slider1 = ttk.Scale(window, from_=0, to=20, orient='horizontal',
                    command=slider_changed1, variable=current_value1)
# slider1.set(15)
slider1.set(13)
slider1.place(x=890, y=50)
value_label1.place(x=995, y=52)


def get_current_value2():
    # print("Erosion value 2: ", current_value2.get())
    return int('{}'.format(current_value2.get()))


def slider_changed2(event2):
    value_label2.configure(text=get_current_value2())


slider_label2 = Label(window, text='Erosion', font=(
    "Times New Roman", 12), fg="black", bg="grey64").place(x=832, y=82)
value_label2 = ttk.Label(window, text=get_current_value2())
slider2 = ttk.Scale(window, from_=0, to=20, orient='horizontal',
                    command=slider_changed2, variable=current_value2)

slider2.set(5)
slider2.place(x=890, y=82)
value_label2.place(x=995, y=82)


# _____________________CREATING BUTTONS______________________

title = Label(window, text="Surveillance System", font=(
    "Times New Roman", 18, 'bold'), fg="black", bg="grey64").place(x=495, y=10)
label_file_explorer = Label(window, text="", fg="blue")
label_file_explorer.place(x=20, y=45)

# _____________________HEADER______________________

title = Label(window, text="Surveillance System", font=(
    "Times New Roman", 18, 'bold'), fg="black", bg="grey64").place(x=495, y=10)
#label_file_explorer = Label(window, text = "", fg = "blue")
# label_file_explorer.place(x=20,y=60)





input_frame = LabelFrame(window.geometry('500x700'), text="Input", font=(
    "Times New Roman", 18, 'bold'), bg="grey64")
input_frame.pack(side='left', expand='yes')

L1 = Label(input_frame, bg="grey64")
L1.pack()

output_frame = LabelFrame(window.geometry('500x700'), text="Output", font=(
    "Times New Roman", 18, 'bold'), bg="grey64")
output_frame.pack(side='right', expand='yes')

L2 = Label(output_frame, bg="grey64")
L2.pack()


displayVar = StringVar()
sample_text_fps = Label(window, bg='grey64', text="FPS: ",
                        font=("Helvetica", 11))
sample_text_fps.place(x=40, y=85)

text_fps = Label(window, bg='grey64', textvariable=displayVar,
                 font=("Helvetica", 11))
text_fps.place(x=75, y=85)

displayVarPath = StringVar()
sample_text_path = Label(window, bg='grey64', text="Path To Output: ",
                        font=("Helvetica", 11))
sample_text_path.place(x=40, y= 145)

text_path=Label(window, bg='grey64', textvariable=displayVarPath,
                 font=("Helvetica", 11))
text_path.place(x=130,y=145)


displayVarFAT = StringVar()
sample_text_fat = Label(window, bg='grey64', text="FAT: ",
                        font=("Helvetica", 11))
sample_text_fat.place(x=40, y=115)

text_fat = Label(window, bg='grey64', textvariable=displayVarFAT,
                 font=("Helvetica", 11))
text_fat.place(x=75, y=115)

# ___________________Old Object detection code___________________

# def objdetect():
#     # source_file = browseFiles()
#     # RealTime data
#     capture = VideoCapture(0)
#     while(1):
#         (ret_old, old_frame) = capture.read()
#         gray_oldframe = cvtColor(old_frame, COLOR_BGR2GRAY)
#         if(is_blur):
#             gray_oldframe = GaussianBlur(gray_oldframe, kernel_gauss, 0)
#         oldBlurMatrix = np.float32(gray_oldframe)
#         accumulateWeighted(gray_oldframe, oldBlurMatrix, 0.003)
#         while(True):
#             ret, frame = capture.read()
#             gray_frame = cvtColor(frame, COLOR_BGR2GRAY)
#             if(is_blur):
#                 newBlur_frame = GaussianBlur(gray_frame, kernel_gauss, 0)
#             else:
#                 newBlur_frame = gray_frame
#             newBlurMatrix = np.float32(newBlur_frame)
#             minusMatrix = absdiff(newBlurMatrix, oldBlurMatrix)
#             ret, minus_frame = threshold(minusMatrix, 60, 255.0, THRESH_BINARY)
#             accumulateWeighted(newBlurMatrix, oldBlurMatrix, 0.02)

#             imshow('Input', frame)

#             drawRectangle(frame, minus_frame)

#             if cv2.waitKey(20) & 0xFF == ord('q'):
#                 break
#         capture.release()
#         cv2.destroyAllWindows()


# def drawRectangle(frame, minus_frame):
#     if(is_blur):
#         minus_frame = GaussianBlur(minus_frame, kernel_gauss, 0)
#     minus_Matrix = np.float32(minus_frame)
#     if(is_close):
#         for i in range(get_current_value1()):
#             minus_Matrix = dilate(minus_Matrix, kernel_d)

#         for i in range(get_current_value2()):
#             minus_Matrix = erode(minus_Matrix, kernel_e)

#     minus_Matrix = np.clip(minus_Matrix, 0, 255)
#     minus_Matrix = np.array(minus_Matrix, np.uint8)
#     contours, hierarchy = findContours(
#         minus_Matrix.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         (x, y, w, h) = boundingRect(c)
#         rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         if(is_draw_ct):
#             drawContours(frame, contours, -1, (0, 255, 255), 1)
#     imshow('Object_Detection', frame)

# def loadVideo(videopath):
#     ImagesSequence = []
#     capture = cv2.VideoCapture(videopath)
#     while(True):
#         ret, frame = capture.read()
#         if ret == True:
#             frame = cv2.flip(frame, 1)
#             ImagesSequence.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
#             cv2.imshow('gray', cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break

#     capture.release()
#     cv2.destroyAllWindows()
#     return ImagesSequence

# ___________________New Object detection code___________________


def loadVideo(videopath):
    global frame_width , frame_height
    ImagesSequence = []
    i = 0
    start = time.time()
    capture = cv2.VideoCapture(videopath)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    while(True):
        ret, frame = capture.read()
        i += 1
        end = time.time()
        if ret == True:
            frame = cv2.flip(frame, 1)
            ImagesSequence.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            cv2.imshow('gray', cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            fps = i/(end-start)
            displayVar.set(str(int(fps)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
    return ImagesSequence


def toggleCapture():
    global capture
    capture = VideoCapture(0)


def write_video(frames_list,fps,type,detur = False):
    if detur:
        Frames_BGR = [cv2.cvtColor(Frame, cv2.COLOR_GRAY2BGR) for Frame in frames_list]
    displayVarPath.set(str(f'{path}/{type}.mp4'))
    out = cv2.VideoWriter(f'{path}/{type}.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, (frames_list[0].shape[1],
                                                                              frames_list[0].shape[0]))
    for i in range(len(frames_list)):
        if detur:
            out.write(Frames_BGR[i].astype(np.uint8))
        else:
            out.write(frames_list[i])
    out.release()


def objdetect():
    global iterFPS
    isVideoCaptureOpen = True
    i = 0
    if keyboard.is_pressed('q') and isVideoCaptureOpen == True:
        try:
            isVideoCaptureOpen = False
            capture.release()
            L1.config(image='')
            L2.config(image='')
            write_video(object_frames, 10, 'object_detect')
            print("Capture released")
            return
        except:
            print("Some error has occured")
    start = time.time()
    (ret_old, old_frame) = capture.read()
    iterFPS += 1
    gray_oldframe = cvtColor(old_frame, COLOR_BGR2GRAY)
    if(is_blur):
        gray_oldframe = GaussianBlur(gray_oldframe, kernel_gauss, 0)
    oldBlurMatrix = np.float32(gray_oldframe)
    accumulateWeighted(gray_oldframe, oldBlurMatrix, 0.003)
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    inpframe = ImageTk.PhotoImage(
        Image.fromarray(cvtColor(frame, COLOR_BGR2RGB)))
    gray_frame = cvtColor(frame, COLOR_BGR2GRAY)
    if(is_blur):
        newBlur_frame = GaussianBlur(gray_frame, kernel_gauss, 0)
    else:
        newBlur_frame = gray_frame

    newBlurMatrix = np.float32(newBlur_frame)
    minusMatrix = absdiff(newBlurMatrix, oldBlurMatrix)
    ret, minus_frame = threshold(minusMatrix, 60, 255.0, THRESH_BINARY)
    accumulateWeighted(newBlurMatrix, oldBlurMatrix, 0.02)

    drawRectangle(inpframe, frame, minus_frame, start, iterFPS)


def drawRectangle(inp_frame, frame, minus_frame, start_time, iterFPS):
    global object_frames
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
    contours, hierarchy = findContours(
        minus_Matrix.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = boundingRect(c)
        rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if(is_draw_ct):
            drawContours(frame, contours, -1, (0, 255, 255), 2)
    end = time.time()
    object_frames.append(frame)
    out_frame = ImageTk.PhotoImage(Image.fromarray(
        cvtColor(frame, cv2.COLOR_BGR2RGB)))
    L1.imgtk = inp_frame
    L1.configure(image=inp_frame)
    L2.imgtk = out_frame
    L2.configure(image=out_frame)
    fps = iterFPS/(end-start_time)
    # displayVar.set(str(int(np.random.randint(18, 23))))
    displayVar.set(str(int(fps)))
    L2.after(1, objdetect)


def deturbulence():
    global ROI_enhanced_arr
    dataType = np.float32
    N_FirstReference = 10
    L = 11
    patch_size = (L, L)  # (y,x) [pixels]. isoplanatic region
    patch_half_size = (
        int((patch_size[0] - 1) / 2), int((patch_size[1] - 1) / 2))
    patches_shift = 1  # when equals to one we get full overlap.
    # (y,x). for each side: up/down/left/right
    registration_interval = (15, 15)
    R = 0.08  # iterativeAverageConstant
    m_lambda0 = 0.55 * 10 ** -6
    m_aperture = 0.06
    m_focal_length = 250 * 10 ** -3
    fno = m_focal_length / m_aperture
    readVideo = 1
    # 3 options: 1. via Lucky region for N_firstRef frames, 2. mean of N_firstRef frames 3. first frame.
    ReferenceInitializationOpt = 2

    ImagesSequence = loadVideo(0)
    ImagesSequence = np.array(ImagesSequence).astype(dataType)
    # roi = selectROI(ImagesSequence[0], resize_factor=2)
    roi = (0, 0, ImagesSequence[0].shape[0], ImagesSequence[0].shape[1])
    print(f"THIS IS ROI: {roi}")
    roi_plate_250 = (1092, 830, 564, 228)
    roi_test = (310, 279, 200, 128)
    if readVideo:
        ROI_coord = roi
    else:
        ROI_coord = roi_plate_250
    ROI_coord = (ROI_coord[1], ROI_coord[0], patch_size[1] * int(ROI_coord[3] / patch_size[1]),
                 patch_size[0] * int(ROI_coord[2] / patch_size[0]))  # now roi[0] - rows!
    ROI_arr = []
    enhancedFrames = []

    if ReferenceInitializationOpt == 1:  # option 1: "Lucky" reference frame.
        # create Reference frame by using "lucky imaging" concept on first N_reference frames.
        FusedPatch = MaxSharpnessFusedPatch([frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
                                             for frame in ImagesSequence[:N_FirstReference]], patch_half_size)
        ReferenceFrame = ImagesSequence[N_FirstReference]
        ReferenceFrame[ROI_coord[0] + patch_half_size[0]:ROI_coord[0] + ROI_coord[2] - patch_half_size[0],
                       ROI_coord[1] + patch_half_size[1]:ROI_coord[1] + ROI_coord[3] - patch_half_size[1]] = FusedPatch
        startRegistrationFrame = N_FirstReference
    # option 2: Mean of N_FirstReference frames.
    elif ReferenceInitializationOpt == 2:
        ReferenceFrame = np.mean(ImagesSequence[:N_FirstReference], axis=0)
        startRegistrationFrame = N_FirstReference
    elif ReferenceInitializationOpt == 3:  # option 3: first frame
        ReferenceFrame = ImagesSequence[0]
        startRegistrationFrame = 1
    else:
        assert Exception("only values 1, 2 or 3 are acceptable")
    enhancedFrames.append(ReferenceFrame)
    i = 0
    for frame in ImagesSequence[startRegistrationFrame:]:
        t = time.time()
        enhancedFrame = np.copy(frame)
        ROI = frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2],
                    ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_arr.append(ROI*255.0/ROI.max())
        no_rows_Cropped_Frame, no_cols_Cropped_Frame = \
            (ROI_coord[2] + 2 * registration_interval[0],
             ROI_coord[3] + 2 * registration_interval[1])

        ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = \
            (1 - R) * ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] + \
            R * frame[ROI_coord[0]: ROI_coord[0] + ROI_coord[2],
                      ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_registered = ReferenceFrame[ROI_coord[0]:ROI_coord[0] +
                                        ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]

        m_lambda0 = 0.55 * 10 ** -6
        m_aperture_diameter = 0.055
        m_focal_length = 250 * 10 ** -3
        fno = m_focal_length / m_aperture_diameter
        ROI_reg_norm = ROI_registered / 255

        k = (2 * np.pi) / m_lambda0
        Io = 1.0
        L = 250
        X = np.arange(-m_aperture_diameter/2,
                      m_aperture_diameter/2, m_aperture_diameter/70)
        Y = X
        XX, YY = np.meshgrid(X, Y)
        AiryDisk = np.zeros(XX.shape)
        # print("SHAPE: ", AiryDisk.shape)
        q = np.sqrt((XX-np.mean(Y)) ** 2 + (YY-np.mean(Y)) ** 2)
        beta = k * m_aperture_diameter * q / 2 / L
        AiryDisk = Io * (2 * j1(beta) / beta) ** 2
        AiryDisk_normalized = AiryDisk/AiryDisk.max()
        deblurredROI_wiener = wiener(ROI_reg_norm, psf=AiryDisk, balance=7)
        deblurredROI = deblurredROI_wiener
        deblurredROI = deblurredROI / deblurredROI.max() * 255.0
        enhancedFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2],
                      ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = np.abs(deblurredROI)
        ROI_enhanced_arr.append(deblurredROI)
        enhancedFrames.append(enhancedFrame)
        displayVarFAT.set(str("{:.3f}".format(time.time() - t)))
        try:
            inp_roi = ImageTk.PhotoImage(
                Image.fromarray(ROI_arr[i].astype(np.uint8)))
            out_roi = ImageTk.PhotoImage(Image.fromarray(
                ROI_enhanced_arr[i].astype(np.uint8)))
        except:
            L1.config(image='')
            L2.config(image='')
            return
        L1['image'] = inp_roi
        L2['image'] = out_roi
       
        window.update()

        if cv2.waitKey(20) & 0xFF == ord('q'):
            L1.config(image='')
            L2.config(image='')
            break
        i += 1
    concatenatedVid = [np.hstack((ROI_arr[i], np.zeros((ROI_arr[0].shape[0], 10)), ROI_enhanced_arr[i])).astype(np.float32) for i in range(len(ROI_arr))]
    write_video(concatenatedVid, 10,'deturbulence',True)
    cv2.destroyAllWindows()
   


def endeturbulence():
    source_file = browseFiles()
    dataType = np.float32
    N_FirstReference = 10
    L = 11
    patch_size = (L, L)
    patch_half_size = (
        int((patch_size[0] - 1) / 2), int((patch_size[1] - 1) / 2))
    patches_shift = 1
    registration_interval = (15, 15)
    R = 0.08
    m_lambda0 = 0.55 * 10 ** -6
    m_aperture = 0.06
    m_focal_length = 250 * 10 ** -3
    fno = m_focal_length / m_aperture
    readVideo = 1
    ReferenceInitializationOpt = 2

    ImagesSequence = loadVideo(0)
    ImagesSequence = np.array(ImagesSequence).astype(dataType)
    roi = selectROI(ImagesSequence[0], resize_factor=2)
    roi_plate_250 = (1092, 830, 564, 228)
    roi_test = (310, 279, 200, 128)
    if readVideo:
        ROI_coord = roi
    else:
        ROI_coord = roi_plate_250

    ROI_coord = (ROI_coord[1], ROI_coord[0], patch_size[1] * int(ROI_coord[3] / patch_size[1]),
                 patch_size[0] * int(ROI_coord[2] / patch_size[0]))  # now roi[0] - rows!

    ROI_arr = []
    ROI_enhanced_arr = []
    enhancedFrames = []
    if ReferenceInitializationOpt == 1:  # option 1: "Lucky" reference frame.
        FusedPatch = MaxSharpnessFusedPatch([frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
                                             for frame in ImagesSequence[:N_FirstReference]], patch_half_size)
        ReferenceFrame = ImagesSequence[N_FirstReference]
        ReferenceFrame[ROI_coord[0] + patch_half_size[0]:ROI_coord[0] + ROI_coord[2] - patch_half_size[0],
                       ROI_coord[1] + patch_half_size[1]:ROI_coord[1] + ROI_coord[3] - patch_half_size[1]] = FusedPatch
        startRegistrationFrame = N_FirstReference
    # option 2: Mean of N_FirstReference frames.
    elif ReferenceInitializationOpt == 2:
        ReferenceFrame = np.mean(ImagesSequence[:N_FirstReference], axis=0)
        startRegistrationFrame = N_FirstReference
    elif ReferenceInitializationOpt == 3:  # option 3: first frame
        ReferenceFrame = ImagesSequence[0]
        startRegistrationFrame = 1
    else:
        assert Exception("only values 1, 2 or 3 are acceptable")

        # showImage(ReferenceFrame.astype(np.uint8))
    enhancedFrames.append(ReferenceFrame)
    i = 0
    for frame in ImagesSequence[startRegistrationFrame:]:
        t = time.time()
        enhancedFrame = np.copy(frame)
        ROI = frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2],
                    ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_arr.append(ROI*255.0/ROI.max())

        # Image Registration via optical flow
        no_rows_Cropped_Frame, no_cols_Cropped_Frame = \
            (ROI_coord[2] + 2 * registration_interval[0],
             ROI_coord[3] + 2 * registration_interval[1])

        u, v = optical_flow_tvl1(
            ReferenceFrame[ROI_coord[0] - registration_interval[0]:ROI_coord[0] + ROI_coord[2] + registration_interval[0],
                           ROI_coord[1] - registration_interval[1]:ROI_coord[1] + ROI_coord[3] + registration_interval[1]],
            enhancedFrame[ROI_coord[0] - registration_interval[0]:ROI_coord[0] + ROI_coord[2] + registration_interval[0],
                          ROI_coord[1] - registration_interval[1]:ROI_coord[1] + ROI_coord[3] + registration_interval[1]],
            attachment=10, tightness=0.3, num_warp=3, num_iter=5, tol=4e-4, prefilter=False)

        row_coords, col_coords = np.meshgrid(np.arange(no_rows_Cropped_Frame), np.arange(no_cols_Cropped_Frame),
                                             indexing='ij')

        warp(enhancedFrame[ROI_coord[0] - registration_interval[0]:ROI_coord[0] + ROI_coord[2] + registration_interval[0],
                           ROI_coord[1] - registration_interval[1]:ROI_coord[1] + ROI_coord[3] + registration_interval[1]],
             np.array([row_coords + v, col_coords + u]), mode='nearest', preserve_range=True).astype(dataType)

        # Iterative averaging ROI
        ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = \
            (1 - R) * ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] + \
            R * frame[ROI_coord[0]: ROI_coord[0] + ROI_coord[2],
                      ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_registered = ReferenceFrame[ROI_coord[0]:ROI_coord[0] +
                                        ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]

        m_lambda0 = 0.55 * 10 ** -6
        m_aperture_diameter = 0.055
        m_focal_length = 250 * 10 ** -3
        fno = m_focal_length / m_aperture_diameter
        ROI_reg_norm = ROI_registered / 255
        k = (2 * np.pi) / m_lambda0  # wavenumber of light in vacuum
        Io = 1.0  # relative intensity
        L = 250  # distance of screen from aperture
        X = np.arange(-m_aperture_diameter/2, m_aperture_diameter /
                      2, m_aperture_diameter/70)  # pupil coordinates
        Y = X
        XX, YY = np.meshgrid(X, Y)
        AiryDisk = np.zeros(XX.shape)
        q = np.sqrt((XX-np.mean(Y)) ** 2 + (YY-np.mean(Y)) ** 2)
        beta = k * m_aperture_diameter * q / 2 / L
        AiryDisk = Io * (2 * j1(beta) / beta) ** 2
        AiryDisk_normalized = AiryDisk/AiryDisk.max()
        deblurredROI_wiener = wiener(ROI_reg_norm, psf=AiryDisk, balance=7)
        deblurredROI = deblurredROI_wiener
        deblurredROI = deblurredROI / deblurredROI.max() * 255.0
        enhancedFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2],
                      ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = np.abs(deblurredROI)
        ROI_enhanced_arr.append(deblurredROI)
        enhancedFrames.append(enhancedFrame)
        frame1 = ImageTk.PhotoImage(
            Image.fromarray(ROI_arr[i].astype(np.uint8)))
        L1['image'] = frame1
        window.update()
        frame2 = ImageTk.PhotoImage(Image.fromarray(
            ROI_enhanced_arr[i].astype(np.uint8)))
        L2['image'] = frame2
        window.update()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        i += 1
    cv2.destroyAllWindows()



def deturbWithObjDetec():
    global Combined_frames
    if keyboard.is_pressed('q'):
        try:
            L1.config(image='')
            L2.config(image='')
            return
        except:
            print("Some error has occured")
    dataType = np.float32
    N_FirstReference = 10
    L = 11
    patch_size = (L, L)  # (y,x) [pixels]. isoplanatic region
    patch_half_size = (
        int((patch_size[0] - 1) / 2), int((patch_size[1] - 1) / 2))
    patches_shift = 1  # when equals to one we get full overlap.
    # (y,x). for each side: up/down/left/right
    registration_interval = (15, 15)
    R = 0.08  # iterativeAverageConstant
    m_lambda0 = 0.55 * 10 ** -6
    m_aperture = 0.06
    m_focal_length = 250 * 10 ** -3
    fno = m_focal_length / m_aperture
    readVideo = 1
    # 3 options: 1. via Lucky region for N_firstRef frames, 2. mean of N_firstRef frames 3. first frame.
    ReferenceInitializationOpt = 2

    ImagesSequence = loadVideo(0)
    ImagesSequence = np.array(ImagesSequence).astype(dataType)
    # roi = selectROI(ImagesSequence[0], resize_factor=2)
    roi = (0, 0, ImagesSequence[0].shape[0], ImagesSequence[0].shape[1])
    # print(f"THIS IS ROI: {roi}")
    roi_plate_250 = (1092, 830, 564, 228)
    roi_test = (310, 279, 200, 128)
    if readVideo:
        ROI_coord = roi
    else:
        ROI_coord = roi_plate_250
    ROI_coord = (ROI_coord[1], ROI_coord[0], patch_size[1] * int(ROI_coord[3] / patch_size[1]),
                 patch_size[0] * int(ROI_coord[2] / patch_size[0]))  # now roi[0] - rows!
    ROI_arr = []
    ROI_enhanced_arr = []
    enhancedFrames = []

    if ReferenceInitializationOpt == 1:  # option 1: "Lucky" reference frame.
        # create Reference frame by using "lucky imaging" concept on first N_reference frames.
        FusedPatch = MaxSharpnessFusedPatch([frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
                                             for frame in ImagesSequence[:N_FirstReference]], patch_half_size)
        ReferenceFrame = ImagesSequence[N_FirstReference]
        ReferenceFrame[ROI_coord[0] + patch_half_size[0]:ROI_coord[0] + ROI_coord[2] - patch_half_size[0],
                       ROI_coord[1] + patch_half_size[1]:ROI_coord[1] + ROI_coord[3] - patch_half_size[1]] = FusedPatch
        startRegistrationFrame = N_FirstReference
    # option 2: Mean of N_FirstReference frames.
    elif ReferenceInitializationOpt == 2:
        ReferenceFrame = np.mean(ImagesSequence[:N_FirstReference], axis=0)
        startRegistrationFrame = N_FirstReference
    elif ReferenceInitializationOpt == 3:  # option 3: first frame
        ReferenceFrame = ImagesSequence[0]
        startRegistrationFrame = 1
    else:
        assert Exception("only values 1, 2 or 3 are acceptable")
    enhancedFrames.append(ReferenceFrame)
    i = 0
    # print("LEN OF IMAGES SEQUENCE : ", len(ImagesSequence))
    for frame in ImagesSequence[startRegistrationFrame:]:
        t = time.time()
        enhancedFrame = np.copy(frame)
        ROI = frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2],
                    ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_arr.append(ROI*255.0/ROI.max())
        no_rows_Cropped_Frame, no_cols_Cropped_Frame = \
            (ROI_coord[2] + 2 * registration_interval[0],
             ROI_coord[3] + 2 * registration_interval[1])

        ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = \
            (1 - R) * ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] + \
            R * frame[ROI_coord[0]: ROI_coord[0] + ROI_coord[2],
                      ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_registered = ReferenceFrame[ROI_coord[0]:ROI_coord[0] +
                                        ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]

        m_lambda0 = 0.55 * 10 ** -6
        m_aperture_diameter = 0.055
        m_focal_length = 250 * 10 ** -3
        fno = m_focal_length / m_aperture_diameter
        ROI_reg_norm = ROI_registered / 255

        k = (2 * np.pi) / m_lambda0
        Io = 1.0
        L = 250
        X = np.arange(-m_aperture_diameter/2,
                      m_aperture_diameter/2, m_aperture_diameter/70)
        Y = X
        XX, YY = np.meshgrid(X, Y)
        AiryDisk = np.zeros(XX.shape)
        # print("SHAPE: ", AiryDisk.shape)
        q = np.sqrt((XX-np.mean(Y)) ** 2 + (YY-np.mean(Y)) ** 2)
        beta = k * m_aperture_diameter * q / 2 / L
        AiryDisk = Io * (2 * j1(beta) / beta) ** 2
        AiryDisk_normalized = AiryDisk/AiryDisk.max()
        deblurredROI_wiener = wiener(ROI_reg_norm, psf=AiryDisk, balance=7)
        deblurredROI = deblurredROI_wiener
        deblurredROI = deblurredROI / deblurredROI.max() * 255.0
        enhancedFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2],
                      ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = np.abs(deblurredROI)
        ROI_enhanced_arr.append(deblurredROI)
        enhancedFrames.append(enhancedFrame)
        print('Frame analysis time: ', time.time() - t)
        displayVarFAT.set(str("{:.3f}".format(time.time() - t)))
        try:
            inp_roi = ImageTk.PhotoImage(
                Image.fromarray(ROI_arr[i].astype(np.uint8)))
            out_roi = ImageTk.PhotoImage(Image.fromarray(
                ROI_enhanced_arr[i].astype(np.uint8)))
        except:
            L1.config(image='')
            L2.config(image='')
            return

        gray_oldframe = ROI_arr[i].astype(np.uint8)
        if(is_blur):
            gray_oldframe = GaussianBlur(gray_oldframe, kernel_gauss, 0)
        oldBlurMatrix = np.float32(gray_oldframe)
        accumulateWeighted(gray_oldframe, oldBlurMatrix, 0.003)

        frame = cv2.flip(ROI_enhanced_arr[i].astype(np.uint8), 1)

        gray_frame = frame

        if(is_blur):
            newBlur_frame = GaussianBlur(gray_frame, kernel_gauss, 0)
        else:
            newBlur_frame = gray_frame

        newBlurMatrix = np.float32(newBlur_frame)
        minusMatrix = absdiff(newBlurMatrix, oldBlurMatrix)
        ret, minus_frame = threshold(minusMatrix, 60, 255.0, THRESH_BINARY)
        accumulateWeighted(newBlurMatrix, oldBlurMatrix, 0.02)

        if(is_blur):
            minus_frame = GaussianBlur(minus_frame, kernel_gauss, 0)
        minus_Matrix = np.float32(minus_frame)
        if(is_close):
            for itr in range(get_current_value1()):
                minus_Matrix = dilate(minus_Matrix, kernel_d)

            for itr in range(get_current_value2()):
                minus_Matrix = erode(minus_Matrix, kernel_e)

        minus_Matrix = np.clip(minus_Matrix, 0, 255)
        minus_Matrix = np.array(minus_Matrix, np.uint8)
        contours, hierarchy = findContours(
            minus_Matrix.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)
        for c in contours:
            (x, y, w, h) = boundingRect(c)
            rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if(is_draw_ct):
                drawContours(frame, contours, -1, (0, 255, 255), 2)

        frame = cv2.flip(frame, 1)
        Combined_frames.append(frame)
        
        #cv2.imshow('frame', frame)
        
        out_frame = ImageTk.PhotoImage(Image.fromarray(frame))
        L1['image'] = inp_roi
        L2['image'] = out_frame
        window.update()
        i += 1
    concatenatedVid = [np.hstack((ROI_arr[i], np.zeros((ROI_arr[0].shape[0], 10)), Combined_frames[i])).astype(np.float32) for i in range(len(ROI_arr))]
    write_video(concatenatedVid,10,'combined',True)
    cv2.destroyAllWindows()
    
   


# _____________________CREATING BUTTONS______________________
C3 = Button(window, text="Object Detection", font=(
    "Times New Roman", 12, 'bold'), command=lambda: [toggleCapture(), objdetect()]).place(x=880, y=10)
C4 = Button(window, text="Turbulence Mitigation", font=(
    "Times New Roman", 12, 'bold'), command=deturbulence).place(x=1090, y=10)
C5 = Button(window, text="Enhanced - TM", font=("Times New Roman",
                                                12, 'bold'), command=endeturbulence).place(x=1090, y=60)
C6 = Button(window, text="TM + Detection", font=("Times New Roman",
                                                 12, 'bold'), command=deturbWithObjDetec).place(x=1280, y=10)


window.state('zoomed')
window.mainloop()


# ____________________END OF PROGRAM______________________
