# Self written functions.

import cv2
import numpy as np
from threading import Thread
import sys
import os
import matplotlib.pyplot as plt  # plot graphs
import time
from imutils.video import FPS

from multiprocessing import Process, Pool, Queue
from multiprocessing.pool import Pool as Pool2
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1
import math  # for sqrt function
import scipy.signal  # for 2d convolution
if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue

# passing I/O to a different thread


class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # instantiates our cv2.videocapture object
        self.stream = cv2.VideoCapture(path)
        self.stopped = False  # if threading process should be stopped
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set,stop it
            if self.stopped:
                return
            # otherwise ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if grabbed boolean is false,then
                # we have reached the end

                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return the next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

# region Create non-daemonic processes


class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.


class MyPool(Pool2):
    Process = NoDaemonProcess
# endregion

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
    Im_resized = cv2.resize(
        Image, (int(no_col/resize_factor), int(no_row/resize_factor)))
    roi = cv2.selectROI('select ROI', Im_resized.astype(np.uint8))
    roi = tuple([i*resize_factor for i in roi])
    return roi


def loadImagesFromDir(path, ImageType):
    """This function loads images with suffix 'ImageType' from directory located in 'path' and returns the
    sequence of the images into an array. Useful for reading video that is stored as frames. """
    # inputs: 'path': full path of the directory.
    #         'ImageType': the suffix of the image.
    # output: Sequence of the images in the directory.
    # Images with different suffix than 'ImageType' will be ignored.
    ImagesSequence = []

    for filename in os.listdir(path):
        fileSuffix = filename[filename.find('.'):]
        if fileSuffix in ImageType:
            img = cv2.imread(path + filename)
            ImagesSequence.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        else:
            continue
    return ImagesSequence


def loadVideo(VideoPath):
    print("loadVideo is called")
    ImagesSequence = []
    cap = cv2.VideoCapture(VideoPath)
    ret, frame = cap.read()
    while ret:
        ImagesSequence.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        ret, frame = cap.read()
    print("loadVideo ended")
    # cap.release()
    return ImagesSequence


def SaveVideoFromFrames(Frames, fps, VideoName):
    '''
    :param Frames: Sequence of Images.
    :param fps: fps to display video.
    :param VideoName: string that indicates video's name.
    :return: Video object.
    '''
    Frames_BGR = [cv2.cvtColor(Frame, cv2.COLOR_GRAY2BGR) for Frame in Frames]
    video = cv2.VideoWriter(os.path.dirname(__file__) + "\\Results\\" + VideoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, (Frames[0].shape[1],
                                                                                                                          Frames[0].shape[0]))
    for i in range(len(Frames)):
        video.write(Frames_BGR[i].astype(np.uint8))
    video.release()


# region Scintillation Removal methods: Pyramids & Registration

def ShowRegistration(image0, image1, warped_image1, dataType):
    # inputs: image0: reference image
    #         image1: image to be warped, before warping
    #         warped_image1: warped image.
    # assumption: image0, image1, warped_image1 are of the same shape.

    # build an RGB image with the unregistered sequence
    no_rows_Cropped_Frame, no_cols_Cropped_Frame = image0.shape
    seq_im = np.zeros(
        (no_rows_Cropped_Frame, no_cols_Cropped_Frame, 3)).astype(dataType)
    seq_im[..., 0] = image1
    seq_im[..., 1] = image1  # image0
    seq_im[..., 2] = image1  # image0

    # build an RGB image with the registered sequence
    reg_im = np.zeros(
        (no_rows_Cropped_Frame, no_cols_Cropped_Frame, 3)).astype(dataType)
    reg_im[..., 0] = warped_image1
    reg_im[..., 1] = image0
    reg_im[..., 2] = image0

    # build an RGB image with the registered sequence
    target_im = np.zeros(
        (no_rows_Cropped_Frame, no_cols_Cropped_Frame, 3)).astype(dataType)
    target_im[..., 0] = image0
    target_im[..., 1] = image0
    target_im[..., 2] = image0

    # --- Show the result

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 10))

    ax0.imshow(seq_im)
    ax0.set_title("Unregistered sequence")
    ax0.set_axis_off()

    ax1.imshow(reg_im)
    ax1.set_title("Registered sequence")
    ax1.set_axis_off()

    ax2.imshow(target_im)
    ax2.set_title("Target")
    ax2.set_axis_off()

    fig.tight_layout()
    plt.show()


def createGradientLaplacianPyramid(I, N):
    ''' Inputs: I - image to be decomposed.
                N - number of levels in pyramid.
    '''
    # set filters:
    d1 = np.array([[1, -1]])
    d2 = np.array([[0, -1], [1, 0]])/math.sqrt(2)
    d3 = np.array([[-1], [1]])
    d4 = np.array([[-1, 0], [0, 1]]) / math.sqrt(2)
    d = [d1, d2, d3, d4]
    w_dot = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    w = scipy.signal.convolve2d(w_dot, w_dot)
    w_new = w + [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

    # cv2.filter2D
    GaussPyramid = [I.copy()]

    # d1 = np.array([[1, -1]])
    # GaussPyramid = [cv2.filter2D(I.copy(), -1, d1)]

    LaplacianPyramid = []

    for i in range(N):
        # filters high-frequencies (noise)
        G_reduced = cv2.pyrDown(GaussPyramid[i])
        GaussPyramid.append(G_reduced)
        D_orientations = [cv2.filter2D(GaussPyramid[i] + cv2.filter2D(GaussPyramid[i], -1, w_dot), -1,
                                       d_k) for d_k in d]
        L_temp = [-1/8 * cv2.filter2D(D_orientations[k], -1, d[k])
                  for k in range(len(d))]
        L_temp_sum = sum(L_temp)
        LaplacianPyramid.append(cv2.filter2D(L_temp_sum, -1, w_new))

    GaussPyramid[-1] = ()
    # Laplace[N] = Gauss[N] -> keep DC data.
    LaplacianPyramid[N-1] = GaussPyramid[N-1]
    return GaussPyramid, LaplacianPyramid


def createGaussianAndLaplacianPyramid(I, N):
    ''' Inputs: I - image to be decomposed.
                N - number of levels in pyramid.
    '''

    GaussPyramid = [I.copy()]
    LaplacianPyramid = []

    for i in range(N):
        # filters high-frequencies (noise)
        G_reduced = cv2.pyrDown(GaussPyramid[i])
        LaplacianPyramid.append(GaussPyramid[i] - cv2.pyrUp(
            G_reduced, dstsize=(GaussPyramid[i].shape[1], GaussPyramid[i].shape[0])))
        GaussPyramid.append(G_reduced)

    GaussPyramid[-1] = ()
    # Laplace[N] = Gauss[N] -> keep DC data.
    LaplacianPyramid[N-1] = GaussPyramid[N-1]
    return GaussPyramid, LaplacianPyramid


def reconstructImageFromLaplacPyramid(laplacianPyr):

    no_of_levels = len(laplacianPyr)
    temp = laplacianPyr[-1]
    for i in range(no_of_levels-2, -1, -1):
        temp = laplacianPyr[i] + cv2.pyrUp(temp, dstsize=(
            laplacianPyr[i].shape[1], laplacianPyr[i].shape[0]))

    return temp


def calculateLaplacianEnergy(Image, patchHalfSize):
    ''' This function calculates the Energy of the laplacian of a given image - chosen by the pyramid and
    the level of the pyramid.
    Returns: Laplacian's Energy of patches & patches.
    '''

    levelSize = Image.shape
    # full overlap between adjacent patches.
    # patchCenterCoordinates = [(row, col) for row in range(patchHalfSize[0], levelSize[0])
    #                           for col in range(patchHalfSize[1], levelSize[1])]
    # patches = [pyramid[level][(row - patchHalfSize[0]):(row + patchHalfSize[0] + 1),
    #            (col - patchHalfSize[1]):(col + patchHalfSize[1] + 1)] for (row, col) in
    #            patchCenterCoordinates]

    # No overlap between adjacent patches.
    patchCenterCoordinates = [(row, col) for row in range(patchHalfSize[0], levelSize[0]) if (row - (
        patchHalfSize[0])) % (patchHalfSize[0] * 2 + 1) == 0
        for col in range(patchHalfSize[1], levelSize[1]) if (col - (patchHalfSize[
            1])) % (patchHalfSize[1] * 2 + 1) == 0]
    patches = [Image[(row - patchHalfSize[0]):(row + patchHalfSize[0] + 1),
                     (col - patchHalfSize[1]):(col + patchHalfSize[1] + 1)] for (row, col) in patchCenterCoordinates]

    # Remove noise by blurring with a Gaussian filter
    # Gaussian_of_patches = [cv2.GaussianBlur(patch, (3, 3), 0) for patch in patches]

    # Salience measurement: max energy of local Laplacian.
    Laplacian_of_patches = [cv2.Laplacian(
        patch, ddepth=cv2.CV_16S, ksize=3) for patch in patches]
    Energy_of_Laplacians = [sum(sum(np.square(patch.astype(np.int64))))
                            for patch in Laplacian_of_patches]
    return Energy_of_Laplacians, patches, patchCenterCoordinates


def combinePyramids(pyramids):
    '''
    Combination of M pyramids based on SELECTIVE mode for image enhancement.
    SELECTIVE mode: selection of most dominant salient measure for each level in the pyramids.
    The salient measure used: Energy of local Laplacian pattern.

    :param pyramids: list of pyramids to be combined.
    :return: one enhanced pyramid.
    '''

    no_of_pyramids = len(pyramids)  # Number of pyramids to be combined.
    no_of_levels = len(pyramids[0])
    # window's size to calculate salient measure. ODD SIZE ONLY
    patchSize = (5, 5)
    patchHalfSize = (int((patchSize[0] - 1) / 2), int((patchSize[1] - 1) / 2))

    combinedPyramid = []

    pool = Pool()

    for level in range(no_of_levels):
        result = pool.starmap(calculateLaplacianEnergy, [(pyramid[level], patchHalfSize)
                                                         for pyramid in pyramids])

        pool.close()
        pool.join()

        LaplacianEnergy_level, patches, patchCenterCoordinates = zip(*result)
        patchCenterCoordinates = patchCenterCoordinates[0]
        #PyramidLaplacian_Max.append(np.max(LaplacianEnergy_level, axis=0))
        MaxLaplacianEnergyPatches_indices = np.argmax(
            LaplacianEnergy_level, axis=0)

        # Initialize new frame as mean of frames rather than zeros. Bottom and right edges of remained zeros will appear as artifacts in the reconstructed image.

        #combinedPyramidLevel = np.mean([pyramid[level] for pyramid in pyramids], axis=0).astype(np.uint8)
        combinedPyramidLevel = pyramids[0][level]

        for counter, max_index in enumerate(MaxLaplacianEnergyPatches_indices):
            combinedPyramidLevel[(patchCenterCoordinates[counter][0] - patchHalfSize[0]):(patchCenterCoordinates[counter][0] + patchHalfSize[0] + 1),
                                 (patchCenterCoordinates[counter][1] - patchHalfSize[1]):(patchCenterCoordinates[counter][1] + patchHalfSize[1] + 1)] = patches[max_index][counter]

        combinedPyramid.append(combinedPyramidLevel)

    return combinedPyramid

    # processes = []
    # for i in range(5):
    #     p = Process(target=f, args=([i for i in range(100)], [100 * level for _ in range(100)]))
    #     p.start()
    #     processes.append(p)
    # for process in processes:
    #     process.join()

    #results = [pool.apply(f, args=(level, level, level + 1)) for level in range(no_of_levels)]
    # pool.close()
    # pool.join()


def combinePyramidsSerial(pyramids):
    no_of_pyramids = len(pyramids)  # Number of pyramids to be combined.
    no_of_levels = len(pyramids[0])
    patchSize = (5, 5)  # window's size to calculate salient measure.
    patchHalfSize = (int((patchSize[0] - 1) / 2), int((patchSize[1] - 1) / 2))
    combinedPyramid = []
    result = []

    for level in range(no_of_levels):
        for pyramid in pyramids:
            result.append(calculateLaplacianEnergy(
                pyramid[level], patchHalfSize))

        LaplacianEnergy_level, patches, patchCenterCoordinates = zip(*result)
        patchCenterCoordinates = patchCenterCoordinates[0]
        # PyramidLaplacian_Max.append(np.max(LaplacianEnergy_level, axis=0))
        MaxLaplacianEnergyPatches_indices = np.argmax(
            LaplacianEnergy_level, axis=0)

        # Initialize combinePyramidLevel as first frame.
        # Otherwise -> Bottom and right edges of remained zeros will appear as artifacts in the reconstructed image.

        #combinedPyramidLevel = np.mean([pyramid[level] for pyramid in pyramids], axis=0).astype(np.uint8)
        combinedPyramidLevel = pyramids[0][level]

        for counter, max_index in enumerate(MaxLaplacianEnergyPatches_indices):
            combinedPyramidLevel[(patchCenterCoordinates[counter][0] - patchHalfSize[0]):(
                patchCenterCoordinates[counter][0] + patchHalfSize[0] + 1),
                (patchCenterCoordinates[counter][1] - patchHalfSize[1]):(
                patchCenterCoordinates[counter][1] + patchHalfSize[1] + 1)] = patches[max_index][counter]

        combinedPyramid.append(combinedPyramidLevel)
        result = []  # Initialize for next level.

    return combinedPyramid

# This function is using most functions listed above.


def RegisterAndEnhanceSequenceOfFrames(ToBeRegisteredSequence, ReferenceFrame, no_frames,
                                       no_of_pyramid_levels, dataType):
    # Image registration: bringing N frames into accurate alignment with reference frame.
    # Registration via optical flow. optical flow is the vector field (u, v) verifying I1(x+u, y+v) = I2(x,y)
    # where I1, I2 are 2D frames from a sequence.
    # TODO: think of the best way to define reference frame.
    no_rows_Cropped_Frame, no_cols_Cropped_Frame = ToBeRegisteredSequence[0].shape
    RegisteredSequence = []

    pool2 = Pool(2)

    result = pool2.starmap(optical_flow_tvl1, [(
        Frame, ReferenceFrame) for Frame in ToBeRegisteredSequence])

    pool2.close()
    pool2.join()

    u, v = zip(*result)

    row_coords, col_coords = np.meshgrid(np.arange(no_rows_Cropped_Frame), np.arange(no_cols_Cropped_Frame),
                                         indexing='ij')
    RegisteredSequence = [warp(ToBeRegisteredSequence[index], np.array([row_coords + v[index], col_coords + u[
        index]]), mode='nearest', preserve_range=True).astype(dataType) for index in range(no_frames)]

    # Show results:
    # frame_no = 2
    # ShowRegistration(ReferenceFrame, ToBeRegisteredSequence[frame_no], RegisteredSequence[frame_no], dataType)

    #print("Image Registration for %d sequent frames took %.2f seconds." % (no_frames, (time.time() - t)))

    ###################################################################################################

    # Image Fusion: Enhancement of each frame via laplacian pyramids algorithm. Method is based on high temporal correlation.
    # Idea: decompose image into N Laplacian levels -> compute weight based on salient pattern at
    # each level -> reconstruct image from level N to level 0.

    # generate pyramids. Pyramids' size is changed to some power of 2. (2^n)
    laplacianPyramids = [createGaussianAndLaplacianPyramid(RegisteredSequence[i], no_of_pyramid_levels)[1]
                         for i in range(len(RegisteredSequence))]

    # t2 = time.time()
    # newPyramid = combinePyramids(laplacianPyramids)
    # print("Image Enhancement - Parallel took: %.2f seconds" % (time.time() - t2))
    # reconstructImage = reconstructImageFromLaplacPyramid(newPyramid)
    # #showImage(reconstructImage.astype(dataType))

    #t3 = time.time()
    newPyramidSerial = combinePyramidsSerial(laplacianPyramids)
    #print("Image Enhancement - Serial took: %.2f seconds" % (time.time() - t3))
    reconstructImage2 = reconstructImageFromLaplacPyramid(newPyramidSerial)
    return reconstructImage2

# endregion

# region methods for DeTurbulence by OTF.


def createPatchIndices(Center, patchSize, ImSize):
    '''
    Creation of indices in vectorized Image for window with size 'patchSize' centered at 'Center' in
    source Image.
    :param Center: Tuple: (y,x) - Center of window.
    :param patchSize: 2x1 vector indicates patch size in each axis. PatchSize is with odd number of rows/cols.
    :param ImSize: 2x1 vector indicates source Image's size. Window is a patch on Image.
    :return: Indices for vectorized Image [N*M-1,1].
    '''
    no_rows_in_Image = ImSize[0]
    no_cols_in_Image = ImSize[1]
    no_rows_in_patch = patchSize[0]
    no_cols_in_patch = patchSize[1]

    patch_rows = np.array(range(
        Center[0]-int((no_rows_in_patch - 1)/2), Center[0]+int((no_rows_in_patch + 1)/2)))
    patch_cols = np.array(range(
        Center[1] - int((no_cols_in_patch - 1) / 2), Center[1] + int((no_cols_in_patch + 1) / 2)))

    # Make sure patch's rows/cols are inside Image.
    remove_indices = np.where((patch_rows < 0) | (
        patch_rows > no_rows_in_Image - 1))
    patch_rows = np.delete(patch_rows, remove_indices)
    remove_indices = np.where((patch_cols < 0) | (
        patch_cols > no_cols_in_Image - 1))
    patch_cols = np.delete(patch_cols, remove_indices)

    windowSize = (max(patch_rows.shape), max(patch_cols.shape))

    patch_rows = np.repeat(patch_rows, windowSize[1])
    patch_cols = np.matlib.repmat(patch_cols, 1, windowSize[0]).reshape(
        windowSize[0]*windowSize[1],)

    indices = no_cols_in_Image * patch_rows + patch_cols

    return indices

# endregion  functions

# region Deturbulence methods


def MaxSharpnessFusedPatch(patches, patch_half_size):
    "Using multiprocessing to calculate simultaniously patches' sharpness metric which is defined as Intensity Variance. "
    no_rows_patch, no_cols_patch = patches[0].shape

    pool = Pool()
    patchesSharpness = pool.starmap(
        MeasureSharpness, [(patch, patch_half_size) for patch in patches])
    pool.close()
    pool.join()

    MaxSharpnessMeasurementsPatches_indices = np.argmax(
        patchesSharpness, axis=0)
    FusedPatch = np.zeros(
        tuple(map(lambda i, j: i - 2*j, patches[0].shape, patch_half_size)))
    no_rows_FusedPatch, no_cols_FusedPatch = FusedPatch.shape

    for counter, maxSharpness_index in enumerate(MaxSharpnessMeasurementsPatches_indices):
        FusedPatch[counter % no_rows_FusedPatch, int(counter / no_rows_FusedPatch)] =\
            patches[maxSharpness_index][patch_half_size[0] + counter %
                                        no_rows_FusedPatch, patch_half_size[1] + int(counter / no_rows_FusedPatch)]

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
    VectorizedImage = patch.reshape(
        Image_size[0]*Image_size[1], 1)  # Image to vector
    Image_mean = np.mean(VectorizedImage)
    Variance = 1/(Image_size[0]*Image_size[1] - 1) * \
        sum((VectorizedImage - Image_mean)**2)

    return int(Variance)


# endregion

#~~~~~~~~~~~~~~~ END OF UTILS ~~~~~~~~~~~~~~~~~~~#
#################################################################################################
