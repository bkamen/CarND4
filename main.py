from camera_calibration import calibrate_camera, undistort_images
from image_transform import image_trafo, image_trafo_folder
import cv2
import pickle

# define camera calibration file
cam_cal = 'camera_calibration_20170525_23h24'


# camera calibration
# get camera calibration values out of function by passing folder containing images, nx, ny and if you want a LOG
# and the result to be saved (will be saved as pickle file, first mtx, then dist)
if cam_cal=='':
    ret, mtx, dist = calibrate_camera(folder='.\camera_cal/', nx=9, ny=6, save_result=0, save_log=0)
else:
    with open(cam_cal, 'rb') as f:  # Python 3: open(..., 'rb')
        mtx, dist = pickle.load(f)


# create examples of undistorted images
#undistort_images('.\camera_cal/', mtx, dist, nx=9, ny=6, chessboarddrawn=1)


# transform images for lane detection
#folder = './test_images/'
#out = image_trafo_folder(folder, mtx, dist, thresh_r=(200, 255), thresh_h=(0, 255), thresh_s=(50, 255),
#                         thresh_sobel=(10, 255), test_saves=0, undistort=1)

file = './test_images/test1.jpg'
img = cv2.imread(file)

out = image_trafo(img, mtx, dist, thresh_r=(200, 255), thresh_g=(195, 255), thresh_h=(0, 255), thresh_s=(100, 255),
                  thresh_sobel=(10, 255), test_saves=1, undistort=1)

cv2.imwrite('test.jpg', out*255)

