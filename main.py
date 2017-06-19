from camera_calibration import calibrate_camera, undistort_images
from image_transform import image_trafo_folder, persp_transform, image_trafo
from fit_lane import init_fit_line, similar_fit_line, meas_curv, draw_lane
import cv2
import pickle
import glob
from moviepy.editor import VideoFileClip
from line import Line

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


folder = './test_images/'
im_list = glob.glob(folder+'*.jpg')

for i in im_list:
    img = cv2.imread(i)
    img, _, _, _, _, _, _, undist, M = image_trafo(img, mtx, dist, thresh_r, thresh_g, thresh_h, thresh_s, thresh_sobel,
                                                   undistort=1, perspective_transform=1)

    #left_fit, right_fit = init_fit_line(img)
    #leftcurv, rightcurv = meas_curv(left_fit,right_fit)

    #img = draw_lane(img, undist, left_fit, right_fit, M)
    #cv2.imshow('transformed image', img)
    #cv2.waitKey(1000)


def img_pipeline(img):
    # undistort and transform image to birds-eye-view
    img, _, _, _, _, _, _, undist, M = image_trafo(img, mtx, dist, thresh_r, thresh_g, thresh_h, thresh_s, thresh_sobel,
                                                   undistort=1, perspective_transform=1)
    # left lane
    llane = Line()
    # right lane
    rlane = Line()

    if not llane.detected:
        init_fit_line(img, llane, rlane)
    else:
        similar_fit_line(img, llane, rlane)
        llane.best_fit = 0.75 * llane.best_fit + 0.25 * llane.current_fit
        rlane.best_fit = 0.75 * rlane.best_fit + 0.25 * rlane.current_fit

#    leftcurv, rightcurv = meas_curv(left_fit, right_fit)

    img = draw_lane(img, undist, llane, rlane, M)
    return img

white_output = 'project_video_output.mp4'
clip1 = VideoFileClip('project_video.mp4', audio=False)
white_clip = clip1.fl_image(img_pipeline)
white_clip.write_videofile(white_output, audio=False)
