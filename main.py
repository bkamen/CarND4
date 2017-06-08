from camera_calibration import calibrate_camera, undistort_images
from image_transform import image_trafo_folder, persp_transform
import cv2
import pickle

# define camera calibration file, if there is any already any
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

thresh_r =    (200, 255)
thresh_g =    (200, 255)
thresh_h =    (160, 255)
thresh_s =    (180, 255)
thresh_sobel = (40, 200)

# find points for perspective transform
file = './test_images/straight_lines1.jpg'
img = cv2.imread(file)
imgout = persp_transform(img)
# define the filename for the output
filename = './output_images/perspective_transformed_' + file.split('/')[-1]
# create the file
cv2.imwrite(filename, imgout)

# transform images for lane detection
folder = './test_images/'
image_trafo_folder(folder, mtx, dist, thresh_r, thresh_g, thresh_h, thresh_s, thresh_sobel,
                   test_saves=1, undistort=1, perspective_transform=1)

# fit the lines and draw them in the image
folder = './output_images/'
fit_spline_folder(folder)