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
from utils import *
import time
from skimage.restoration import wiener, richardson_lucy
from scipy.special import j1


#_____________________USER-DEFINED FUNCTIONS______________________

kernel_d = np.ones((3,3), np.uint8)
kernel_e = np.ones((3,3), np.uint8)
kernel_gauss = (3,3)
is_blur = True                          #initializing_boolean_variables
is_close = True                         #initializing_boolean_variables
is_draw_ct = False                      #initializing_boolean_variables
fac = 2                                 #initializing_integer_variables
i=1
#___________________INITALIZING THE GUI WINDOW______________________

   
window=Tk()
window.configure(background="grey64");
window.title("Surveillance System")
window.resizable(0,0)
window.geometry('1300x680')

#Need to setup GUI to maximum screen size automatically and adjust widgets accordingly

#_______________SETTING VARIBALES TO CHECK STATE OF BUTTON (CHECKED OR UNCHECKED)______________________


current_value1 = IntVar()
current_value2 = IntVar()


def get_current_value1():
    return int('{}'.format(current_value1.get()))

def slider_changed1(event):
    value_label1.configure(text=get_current_value1())

slider_label1 = Label(window,text='Dilation',font=("Times New Roman",12),fg="black",bg="grey64").place(x=832,y=52)
value_label1 = ttk.Label(window, text=get_current_value1())
slider1 = ttk.Scale(window, from_=5,to=25, orient='horizontal', command=slider_changed1, variable=current_value1)
slider1.set(15)
slider1.place(x=890,y=50)
value_label1.place(x=995,y=52)

#Need to pass value of slider dynamically to objdetect()

def get_current_value2():
    return int('{}'.format(current_value2.get()))

def slider_changed2(event2):
    value_label2.configure(text=get_current_value2())

slider_label2 = Label(window,text='Erosion',font=("Times New Roman",12),fg="black",bg="grey64").place(x=832,y=82)
value_label2 = ttk.Label(window, text=get_current_value2())
slider2 = ttk.Scale(window, from_=5,to=25, orient='horizontal', command=slider_changed2, variable=current_value2)
slider2.set(15)
slider2.place(x=890,y=82)
value_label2.place(x=995,y=82)


#_____________________CREATING BUTTONS______________________

title = Label(window, text = "Surveillance System",font=("Times New Roman",18, 'bold'),fg="black",bg="grey64").place(x=495, y=10)
label_file_explorer = Label(window, text = "", fg = "blue")
label_file_explorer.grid(column = 1, row = 1)

#____________________ADDING FUNCTIONALITES_________________________

def browseFiles():
   global source_file
   source_file = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes =[('All Files', '.*')],parent=window)
   label_file_explorer.configure(text=""+source_file)
   return source_file

def live():
    def objdetect():
        capture = VideoCapture(0)
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
   
#Need to fit the  imshow('Object_Detection', frame) inside the GUI (instead of pop up)

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
	imshow('Object_Detection', frame)

def objdetect():
	
	capture = VideoCapture(source_file);
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


def deturbulence():
    dataType = np.float32
    N_FirstReference = 10
    L = 11
    patch_size = (L, L)
    patch_half_size = (int((patch_size[0] - 1) / 2), int((patch_size[1] - 1) / 2))
    patches_shift = 1
    registration_interval = (15, 15)  # (y,x). for each side: up/down/left/right
    R = 0.08
    m_lambda0 = 0.55 * 10 ** -6
    m_aperture = 0.06
    m_focal_length = 250 * 10 ** -3
    fno = m_focal_length / m_aperture
        #m_Fried_no = m_aperture / 6

        ## define Flags
    readVideo = 1  # flag indicating whether video is read or a sequence of images is loaded.
    ReferenceInitializationOpt = 2 # 3 options: 1. via Lucky region for N_firstRef frames, 2. mean of N_firstRef frames 3. first frame.

    video_path = source_file
    ImagesSequence = loadVideo(video_path)
    ImagesSequence = np.array(ImagesSequence).astype(dataType)
        # endregion

        # region Define ROI

        # resize image for ROI choosing so it will fit in screen.
    roi = selectROI(ImagesSequence[0], resize_factor=2)

    roi_plate_250 = (1092, 830, 564, 228)
    roi_test = (310, 279, 200, 128)
        #roi_FULLvid = (0, 0, ImagesSequence[0].shape[1], ImagesSequence[0].shape[0])

    if readVideo:
        ROI_coord = roi
    else:
        ROI_coord = roi_plate_250 # roi[0] - column, roi[1] - rows. will be inverted for convenience.

        # Make sure roi dimensions are a multiplication of patch_size & make first dimension as rows index.
    ROI_coord = (ROI_coord[1], ROI_coord[0], patch_size[1] * int(ROI_coord[3] / patch_size[1]),
                     patch_size[0] * int(ROI_coord[2] / patch_size[0]))  # now roi[0] - rows!

        # patchCenterCoordinates = [(row, col) for row in
        #                           range(roi[0] + patch_half_size[0], roi[0] + roi[2] - patch_half_size[0]) if
        #                           (row - roi[0]) % patches_shift == 0
        #                           for col in range(roi[1] + patch_half_size[1], roi[1] + roi[3] - patch_half_size[1]) if
        #                           (col - roi[1]) % patches_shift == 0]

        # endregion

        ###################################################################################################
        ######### RUN ALGORITHM: Every iteration consists 3 main stages:
        #                - step 1: Image Registration via optical flow: pixel-wise.
        #                - step 2: patches fusion: iterative averaging patches
        #                - step 3: roi deconvolution.

        # Step 1: define Reference frame. (3 options: 1. via Lucky region for N_firstRef frames, 2. mean of N_firstRef frames
        # 3. first frame.
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

        #showImage(ReferenceFrame.astype(np.uint8))
    enhancedFrames.append(ReferenceFrame)

    for frame in ImagesSequence[startRegistrationFrame:]:
        t = time.time()
        enhancedFrame = np.copy(frame)
        ROI = frame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]]
        ROI_arr.append(ROI*255.0/ROI.max())

            ## Image Registration via optical flow
        no_rows_Cropped_Frame, no_cols_Cropped_Frame = \
                (ROI_coord[2] + 2 * registration_interval[0], ROI_coord[3] + 2 * registration_interval[1])

            # if no_rows_Cropped_Frame > (ReferenceFrame.shape[0] - registration_interval[0]):
            #     no_rows_Cropped_Frame = ReferenceFrame.shape[0] - registration_interval[0] - 1
            #
            # if no_cols_Cropped_Frame > ReferenceFrame.shape[1] - 2*registration_interval[0]:
            #     no_cols_Cropped_Frame = ReferenceFrame.shape[1] - 2*registration_interval[1]
            # TODO: check whether patch-wise OF gets better results or is faster.
            # TODO: find best values of parameters based on metric evaluation.
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
            ## Deconvolution of ROI
            # # blind Richardson-Lucy deconvolution
            # ROI_normalized = ROI / 255
            # deblurredROI_blind, BlindPsf_evaluated = richardson_lucy(ROI_normalized)
            # deblurredROI_blind = deblurredROI_blind * 255.0 / deblurredROI_blind.max()
            # non-blind deconvolution
        m_lambda0 = 0.55 * 10 ** -6
        m_aperture_diameter = 0.055
        m_focal_length = 250 * 10 ** -3
        fno = m_focal_length / m_aperture_diameter
        ROI_reg_norm = ROI_registered / 255
            # Diffraction-limited psf de-blurring
            # Airy disk creation
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
           #deblurredROI_Lucy, psf_evaluated = richardson_lucy(ROI_normalized, psf=AiryDisk,  iterations=70)
        deblurredROI_wiener = wiener(ROI_reg_norm, psf=AiryDisk, balance=7) # ROI should be normalized and AirdDisk shouldn't?
        #plt.figure(figsize=(6.4*3, 4.8*3))
        #plt.subplot(131), plt.imshow(ROI, 'gray')
        #plt.subplot(132), plt.imshow(deblurredROI_Lucy, 'gray')
        #plt.subplot(133), plt.imshow(deblurredROI_wiener, 'gray')
        deblurredROI = deblurredROI_wiener
        deblurredROI = deblurredROI / deblurredROI.max() * 255.0
        enhancedFrame[ROI_coord[0]:ROI_coord[0] + ROI_coord[2], ROI_coord[1]:ROI_coord[1] + ROI_coord[3]] = np.abs(deblurredROI)
        ROI_enhanced_arr.append(deblurredROI)
        enhancedFrames.append(enhancedFrame)
        print('Frame analysis time: ', time.time() - t)
        # save loaded video with ROI enhanced.
        #SaveVideoFromFrames(enhancedFrames, 25.0, 'test_video.avi')
        # save ROIs' difference video (close look).
    concatenatedVid = [np.hstack((ROI_arr[i], np.zeros((ROI_arr[0].shape[0], 10)), ROI_enhanced_arr[i])).astype(np.float32) for i in range(len(ROI_arr))]
    SaveVideoFromFrames(concatenatedVid, 25.0, 'Comparison_testVideoWiener_MeanRef.avi')
        # allowing multiprocessing
    # balance = np.array([0.5, 1, 2, 5, 6, 7, 8, 10, 12, 15, 25, 50])
    # for b in balance:
    #     deblurredROI_wiener = wiener(ROI_normalized, psf=AiryDisk, balance=b)
    #     deblurredROI_wiener = 255.0 * deblurredROI_wiener / deblurredROI_wiener.max()
    #     plt.figure(figsize=(2 * 3, 2 * 2.25))
    #     plt.subplot(121), plt.imshow(ROI, 'gray'), plt.title('ROI')
    #     plt.subplot(122), plt.imshow(deblurredROI_wiener, 'gray'), plt.title('Wiener b=%d' %b)

C1=Button(window,text = "Browse",font=("Times New Roman",12, 'bold'),command=browseFiles).place(x=100,y=10)
C2=Button(window,text="Live Input",font=("Times New Roman",12, 'bold'),command=live).place(x=300,y=10)
C3=Button(window,text = "Object Detection",font=("Times New Roman",12, 'bold'),command=objdetect).place(x=880,y=10)
C4=Button(window,text="Turbulence Mitigation",font=("Times New Roman",12, 'bold'),command=deturbulence).place(x=1090,y=10)

#___________________FOOTER OF THE GUI WINDOW______________________

frame=LabelFrame(window,width=1300, height=50,fg="black",bg="aqua").place(x=0,y=630)
foot=Label(frame,text = "Surveillance System",font=("Times New Roman",11),fg="black",bg="aqua").place(x=1010,y=645)
window.mainloop()

#____________________END OF PROGRAM______________________
