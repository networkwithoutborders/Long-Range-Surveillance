# Self written functions.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt # plot graphs

from skimage.transform import warp
from skimage.registration import optical_flow_tvl1
import math # for sqrt function
import scipy.signal # for 2d convolution


# region Images methods
def showImage(I):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', I)
    cv2.waitKey(0)  # (this is necessary to avoid Python kernel form crashing)
    # closing all open windows
    cv2.destroyAllWindows()

def selectROI(Image, resize_factor):
    "ROI selection with resizing Image so it will fit in screen"
    no_row, no_col = Image.shape
    Im_resized = cv2.resize(Image, (int(no_col/resize_factor), int(no_row/resize_factor)))
    roi = cv2.selectROI('select ROI', Im_resized.astype(np.uint8))
    roi = tuple([i*resize_factor for i in roi])
    return roi

def loadVideo(VidPath):
    ImagesSequence = []
    cap = cv2.VideoCapture(VidPath)
    while(1):
        ret, frame = cap.read()
        if ret == True:
            ImagesSequence.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            
    cap.release()
    cv2.destroyAllWindows()

    return ImagesSequence

def SaveVideoFromFrames(Frames, fps, VideoName):
    '''
    :param Frames: Sequence of Images.
    :param fps: fps to display video.
    :param VideoName: string that indicates video's name.
    :return: Video object.
    '''
    Frames_BGR = [cv2.cvtColor(Frame, cv2.COLOR_GRAY2BGR) for Frame in Frames]
    video = cv2.VideoWriter(os.path.dirname(__file__) +"\\Results\\" + VideoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, (Frames[0].shape[1],
                                                                              Frames[0].shape[0]))
    for i in range(len(Frames)):
        video.write(Frames_BGR[i].astype(np.uint8))
    video.release()


#~~~~~~~~~~~~~~~ END OF UTILS ~~~~~~~~~~~~~~~~~~~#
#################################################################################################

