import numpy as np
import cv2
import glob
import pickle
import time


def calibrate_camera(folder, nx, ny, save_result=1, save_log=1):
    # list of images that should be processed
    im_list = glob.glob(folder+'*.jpg')

    # initialise arrays
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # object points for each image
    objpts = []  # empty array for all object points
    imgpts = []  # empty array for all image points
    cal_successful = np.zeros(len(im_list))  # array for check if detecting corners was successful, needed for log file


    # loop through each image and create a an image in the ouput folder name output images
    for i in range(len(im_list)):
        im = cv2.imread(im_list[i])
        # call function to detect chessboard corners
        ret, corners = corner_detection(im, nx, ny)
        # grayscale conversion for the image shape of the calibrate camera function
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        if ret:
            cal_successful[i] = 1
            objpts.append(objp)
            imgpts.append(corners)

            # Uncomment block to see images with corners drawn on
            # im = cv2.drawChessboardCorners(im, (nx, ny), corners, ret)
            # cv2.imshow('img', im)
            # cv2.waitKey(500)

    # camera calibration function
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)

    # if not elsewise specified save camera calibration output
    if save_result:
        filename = 'camera_calibration_'+time.strftime("%Y%m%d_%Hh%M")
        with open(filename, 'wb') as pickle_file:
            pickle.dump([mtx, dist], pickle_file)

    # write calibration log
    if save_log:
        camera_calibration_log(im_list, cal_successful)

    return ret, mtx, dist


def camera_calibration_log(im_list, cal_successful):
    filename = 'LOG_camera_calibration_'+time.strftime("%Y%m%d_%Hh%M")+'.txt'
    target = open(filename,'w')
    target.truncate()

    target.write('Protocol of camera calibration on ' + time.strftime('%d/%m/%Y') + 'at' + time.strftime('%H:%M'))
    target.write('\n')
    target.write('\n')

    for i in range(len(cal_successful)):
        if cal_successful[i] == 1:
            stri = ' was successful'
        else:
            stri = ' failed'
        target.write('Corner detection on file '+im_list[i]+stri)
        target.write('\n')

    target.close()
    return None


def corner_detection(img, nx, ny):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find the chessboard corners, number of corners passed as parameters
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    return ret, corners


def undistort_images(folder, mtx, dist, nx, ny, chessboarddrawn=1):
    # list of images that should be processed
    im_list = glob.glob(folder + '*.jpg')

    # loop through image list
    for i in range(len(im_list)):
        img = cv2.imread(im_list[i])
        # if user specified that chess board corners should be drawn, the saved file will have it when successfully
        # detected
        if chessboarddrawn:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # function to undistort image with output of camera calibration
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        # define the filename for the output
        filename = './output_images/undistorted_'+im_list[i].split('\\')[-1]
        # create the file
        cv2.imwrite(filename, dst)
    return None
