#_____________________HEADER FILES______________________

import tkinter
from tkinter import*
from tkinter import ttk, Button
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
from utils import *
import time
from skimage.restoration import wiener, richardson_lucy
from scipy.special import j1
import asyncio
from multiprocessing import Process
#_____________________USER-DEFINED FUNCTIONS______________________

kernel_d = np.ones((3,3), np.uint8)
kernel_e = np.ones((3,3), np.uint8)
kernel_gauss = (3,3)
is_blur = False                        #initializing_boolean_variables
is_close = True                         #initializing_boolean_variables
is_draw_ct = False                      #initializing_boolean_variables
fac = 2                                 #initializing_integer_variables

#___________________INITALIZING THE GUI WINDOW______________________

window = Tk()
window.geometry('1500x780')
window.configure(background="grey64");
window.title("Surveillance System")
window.resizable(0,0)

#___________________HEADER OF THE GUI WINDOW______________________

hframe=LabelFrame(window,width=1500, height=50,fg="black",bg="aqua").place(x=0,y=0)
title = Label(hframe, text = "Surveillance System",font=("Times New Roman",18, 'bold'),fg="black",bg="aqua").place(x=680, y=2)


#_______________SETTING VARIBALES TO CHECK STATE OF BUTTON (CHECKED OR UNCHECKED)______________________


current_value1 = IntVar()
current_value2 = IntVar()


def get_current_value1():
    return int('{}'.format(current_value1.get()))

def slider_changed1(event):
    value_label1.configure(text=get_current_value1())

slider_label1 = Label(window,text='Dilation',font=("Times New Roman",12),fg="black",bg="grey64").place(x=1032,y=52)
value_label1 = ttk.Label(window, text=get_current_value1())
slider1 = ttk.Scale(window, from_=5,to=25, orient='horizontal', command=slider_changed1, variable=current_value1)
slider1.set(15)
slider1.place(x=1090,y=50)
value_label1.place(x=1095,y=52)


def get_current_value2():
    return int('{}'.format(current_value2.get()))

def slider_changed2(event2):
    value_label2.configure(text=get_current_value2())

slider_label2 = Label(window,text='Erosion',font=("Times New Roman",12),fg="black",bg="grey64").place(x=1032,y=82)
value_label2 = ttk.Label(window, text=get_current_value2())
slider2 = ttk.Scale(window, from_=5,to=25, orient='horizontal', command=slider_changed2, variable=current_value2)
slider2.set(15)
slider2.place(x=1090,y=82)
value_label2.place(x=1095,y=82)

#____________________ADDING FUNCTIONALITES_________________________
def selectROI(Image, resize_factor):
    "ROI selection with resizing Image so it will fit in screen"
    no_row, no_col = Image.shape
    Im_resized = cv2.resize(Image, (int(no_col/resize_factor), int(no_row/resize_factor)))
    roi = cv2.selectROI('select ROI', Im_resized.astype(np.uint8))
    roi = tuple([i*resize_factor for i in roi])
    return roi


def loadVideo(videopath):
    ImagesSequence = []
    cap = cv2.VideoCapture(videopath)
    while(True):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame,1)
            ImagesSequence.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            cv2.imshow('gray',cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
               
    cap.release()
    cv2.destroyAllWindows()
    return ImagesSequence
    


def MaxSharpnessFusedPatch(patches, patch_half_size):
    "Using multiprocessing to calculate simultaniously patches' sharpness metric which is defined as Intensity Variance. "
    no_rows_patch, no_cols_patch = patches[0].shape

    pool = Pool()
    patchesSharpness = pool.starmap(MeasureSharpness, [(patch, patch_half_size) for patch in patches])
    pool.close()
    pool.join()

    MaxSharpnessMeasurementsPatches_indices = np.argmax(patchesSharpness, axis=0)
    FusedPatch = np.zeros(tuple(map(lambda i, j: i - 2*j, patches[0].shape, patch_half_size)))
    no_rows_FusedPatch, no_cols_FusedPatch = FusedPatch.shape

    for counter, maxSharpness_index in enumerate(MaxSharpnessMeasurementsPatches_indices):
        FusedPatch[counter % no_rows_FusedPatch, int(counter / no_rows_FusedPatch)] =\
            patches[maxSharpness_index][patch_half_size[0] + counter % no_rows_FusedPatch, patch_half_size[1] + int(counter / no_rows_FusedPatch)]

    return FusedPatch

def MeasureSharpness(Image, patchHalfSize):
    ''' This function calculates the sharpness (salience) measurement of a given image.
    It divides the image into patches for a given patch's size. Patches calculation order is determined vertically (changing rows).
    Returns: List of Sharpness Measurement of patches.
    '''

    # full overlap between adjacent patches.
    ROI_size = Image.shape
    patchCenterCoordinates = [(row, col) for row in range(patchHalfSize[0], ROI_size[0] - patchHalfSize[0])
                              for col in range(patchHalfSize[1], ROI_size[1] - patchHalfSize[1])]

    patches = [Image[(row - patchHalfSize[0]):(row + patchHalfSize[0] + 1),
               (col - patchHalfSize[1]):(col + patchHalfSize[1] + 1)] for (row, col) in patchCenterCoordinates]

    # Remove noise by blurring with a Gaussian filter
    #Gaussian_of_patches = [cv2.GaussianBlur(patch, (3, 3), 0) for patch in patches]

    # #Salience measurement : max energy of local Laplacian.
    # Laplacian_of_patches = [cv2.Laplacian(patch, ddepth=cv2.CV_16S, ksize=3) for patch in patches]
    # Energy_of_Laplacians = [sum(sum(np.square(patch.astype(np.int64)))) for patch in Laplacian_of_patches]

    # Sharpness measurement: Intensity variance
    varianceOfPatches = [IntensityVariance(patch) for patch in patches]

    # Found to be slower using pool!
    # pool = Pool(2)
    # varianceOfPatches = pool.map(IntensityVariance, patches)
    # pool.close()
    # pool.join()

    sharpnessValues = varianceOfPatches
    return sharpnessValues

def IntensityVariance(patch):
    " Calculate Variance of patches' intensity."
    Image_size = patch.shape
    VectorizedImage = patch.reshape(Image_size[0]*Image_size[1], 1) #Image to vector
    Image_mean = np.mean(VectorizedImage)
    Variance = 1/(Image_size[0]*Image_size[1] - 1) * sum((VectorizedImage - Image_mean)**2)

    return int(Variance)


L1 = Label(window,height=360,width=360,bg="grey64",padx=0,pady=0)
L1.place(x=250,y=180)

L2 = Label(window,height=360,width=360,bg="grey64",padx=0,pady=0)
L2.place(x=620,y=180)

L3 = Label(window,height=360,width=360,bg="grey64",padx=0,pady=0)
L3.place(x=1000,y=180)

cap = cv2.VideoCapture(0)

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



def objdetect():
    while(1):
        (ret_old, old_frame) = cap.read()
        gray_oldframe = cvtColor(old_frame, COLOR_BGR2GRAY)
        if(is_blur):
            gray_oldframe = GaussianBlur(gray_oldframe, kernel_gauss, 0)
        oldBlurMatrix = np.float32(gray_oldframe)
        accumulateWeighted(gray_oldframe, oldBlurMatrix, 0.003)
        while True:
            ret, frame = cap.read()
            gray_frame = cvtColor(frame, COLOR_BGR2GRAY)
            if(is_blur):
                newBlur_frame = GaussianBlur(gray_frame, kernel_gauss, 0)
            else:
                newBlur_frame = gray_frame
        
            newBlurMatrix = np.float32(newBlur_frame)
            minusMatrix = absdiff(newBlurMatrix, oldBlurMatrix)
            ret, minus_frame = threshold(minusMatrix, 60, 255.0, THRESH_BINARY)
            accumulateWeighted(newBlurMatrix,oldBlurMatrix,0.02)
    
            drawRectangle(frame, minus_frame)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)
            frame = ImageTk.PhotoImage(Image.fromarray(frame))
            L3['image'] = frame
            window.update()
    
    


def deturbulence():
    dataType = np.float32
    N_FirstReference = 10
    L = 11
    patch_size = (L, L)  # (y,x) [pixels]. isoplanatic region
    patch_half_size = (int((patch_size[0] - 1) / 2), int((patch_size[1] - 1) / 2))
    patches_shift = 1  # when equals to one we get full overlap.
    registration_interval = (15, 15)  # (y,x). for each side: up/down/left/right
    R = 0.08  # iterativeAverageConstant
    m_lambda0 = 0.55 * 10 ** -6
    m_aperture = 0.06
    m_focal_length = 250 * 10 ** -3
    fno = m_focal_length / m_aperture
    readVideo = 1  
    ReferenceInitializationOpt = 2 # 3 options: 1. via Lucky region for N_firstRef frames, 2. mean of N_firstRef frames 3. first frame.

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

    if ReferenceInitializationOpt == 1: ## option 1: "Lucky" reference frame.
            # create Reference frame by using "lucky imaging" concept on first N_reference frames.
        FusedPatch = MaxSharpnessFusedPatch([frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] \
                                                 for frame in ImagesSequence[:N_FirstReference]], patch_half_size)
        ReferenceFrame = ImagesSequence[N_FirstReference]
        ReferenceFrame[ROI_coord[0] + patch_half_size[0]:ROI_coord[0] + ROI_coord[2] - patch_half_size[0],
        ROI_coord[1] + patch_half_size[1]:ROI_coord[1] + ROI_coord[3] - patch_half_size[1]] = FusedPatch
        startRegistrationFrame = N_FirstReference
    elif ReferenceInitializationOpt == 2: ## option 2: Mean of N_FirstReference frames.
        ReferenceFrame = np.mean(ImagesSequence[:N_FirstReference], axis=0)
        startRegistrationFrame = N_FirstReference
    elif ReferenceInitializationOpt == 3:  ## option 3: first frame
        ReferenceFrame = ImagesSequence[0]
        startRegistrationFrame = 1
    else:
        assert Exception("only values 1, 2 or 3 are acceptable")
    enhancedFrames.append(ReferenceFrame)
    i=0
    for frame in ImagesSequence[startRegistrationFrame:]:
        t = time.time()
        enhancedFrame = np.copy(frame)
        ROI = frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_arr.append(ROI*255.0/ROI.max())
        no_rows_Cropped_Frame, no_cols_Cropped_Frame = \
                (ROI_coord[2] + 2 * registration_interval[0], ROI_coord[3] + 2 * registration_interval[1])

        ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = \
                (1 - R) * ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] + \
                R * frame[ROI_coord[0]: ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_registered = ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]

        m_lambda0 = 0.55 * 10 ** -6
        m_aperture_diameter = 0.055
        m_focal_length = 250 * 10 ** -3
        fno = m_focal_length / m_aperture_diameter
        ROI_reg_norm = ROI_registered / 255


        k = (2 * np.pi) / m_lambda0
        Io= 1.0
        L= 250 
        X = np.arange(-m_aperture_diameter/2, m_aperture_diameter/2, m_aperture_diameter/70)
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
        enhancedFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = np.abs(deblurredROI)
        ROI_enhanced_arr.append(deblurredROI)
        enhancedFrames.append(enhancedFrame)
        print('Frame analysis time: ', time.time() - t)
        #cv2.imshow('Input',ROI_arr[i].astype(np.uint8))
        frame1 = ImageTk.PhotoImage(Image.fromarray(ROI_arr[i].astype(np.uint8)))
        L1['image'] = frame1
        window.update()
        #cv2.imshow('Output',ROI_enhanced_arr[i].astype(np.uint8))
        frame2 = ImageTk.PhotoImage(Image.fromarray(ROI_enhanced_arr[i].astype(np.uint8)))
        L2['image'] = frame2
        window.update()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        i+=1
    #cv2.destroyAllWindows()  '''


def endeturbulence():
    dataType = np.float32
    N_FirstReference = 10
    L = 11
    patch_size = (L, L)
    patch_half_size = (int((patch_size[0] - 1) / 2), int((patch_size[1] - 1) / 2))
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
    if ReferenceInitializationOpt == 1: ## option 1: "Lucky" reference frame.
        FusedPatch = MaxSharpnessFusedPatch([frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] \
                                                 for frame in ImagesSequence[:N_FirstReference]], patch_half_size)
        ReferenceFrame = ImagesSequence[N_FirstReference]
        ReferenceFrame[ROI_coord[0] + patch_half_size[0]:ROI_coord[0] + ROI_coord[2] - patch_half_size[0],
        ROI_coord[1] + patch_half_size[1]:ROI_coord[1] + ROI_coord[3] - patch_half_size[1]] = FusedPatch
        startRegistrationFrame = N_FirstReference
    elif ReferenceInitializationOpt == 2: ## option 2: Mean of N_FirstReference frames.
        ReferenceFrame = np.mean(ImagesSequence[:N_FirstReference], axis=0)
        startRegistrationFrame = N_FirstReference
    elif ReferenceInitializationOpt == 3:  ## option 3: first frame
        ReferenceFrame = ImagesSequence[0]
        startRegistrationFrame = 1
    else:
        assert Exception("only values 1, 2 or 3 are acceptable")

        #showImage(ReferenceFrame.astype(np.uint8))
    enhancedFrames.append(ReferenceFrame)
    i=0
    for frame in ImagesSequence[startRegistrationFrame:]:
        t = time.time()
        enhancedFrame = np.copy(frame)
        ROI = frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_arr.append(ROI*255.0/ROI.max())

            ## Image Registration via optical flow
        no_rows_Cropped_Frame, no_cols_Cropped_Frame = \
                (ROI_coord[2] + 2 * registration_interval[0], ROI_coord[3] + 2 * registration_interval[1])

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

            ## Iterative averaging ROI
        ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = \
                (1 - R) * ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] + \
                R * frame[ROI_coord[0]: ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_registered = ReferenceFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]

        m_lambda0 = 0.55 * 10 ** -6
        m_aperture_diameter = 0.055
        m_focal_length = 250 * 10 ** -3
        fno = m_focal_length / m_aperture_diameter
        ROI_reg_norm = ROI_registered / 255
        k = (2 * np.pi) / m_lambda0 # wavenumber of light in vacuum
        Io= 1.0 # relative intensity
        L= 250 # distance of screen from aperture
        X = np.arange(-m_aperture_diameter/2, m_aperture_diameter/2, m_aperture_diameter/70) #pupil coordinates
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
        enhancedFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = np.abs(deblurredROI)
        ROI_enhanced_arr.append(deblurredROI)
        enhancedFrames.append(enhancedFrame)
        print('Frame analysis time: ', time.time() - t)
        #cv2.imshow('Input',ROI_arr[i].astype(np.uint8))
        frame1 = ImageTk.PhotoImage(Image.fromarray(ROI_arr[i].astype(np.uint8)))
        L1['image'] = frame1
        window.update()
        #cv2.imshow('Output',ROI_enhanced_arr[i].astype(np.uint8))
        frame2 = ImageTk.PhotoImage(Image.fromarray(ROI_enhanced_arr[i].astype(np.uint8)))
        L2['image'] = frame2
        window.update()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        i+=1
    cv2.destroyAllWindows()


C3=Button(window,text = "Object Detection",font=("Times New Roman",12, 'bold'), command=objdetect).place(x=20,y=60)
C4=Button(window,text="Turbulence Mitigation",font=("Times New Roman",12, 'bold'),command=deturbulence).place(x=20,y=100)
C5=Button(window,text="Enhanced - TM",font=("Times New Roman",12, 'bold'),command=endeturbulence).place(x=20,y=140)

#___________________FOOTER OF THE GUI WINDOW______________________

frame=LabelFrame(window,width=1500, height=50,fg="black",bg="aqua").place(x=0,y=720)
foot=Label(frame,text = "Developed using Python 3.8",font=("Times New Roman",11),fg="black",bg="aqua").place(x=2,y=730)
window.mainloop()
 

#____________________END OF PROGRAM______________________
