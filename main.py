from camera_calibration import calibrate_camera
from image_transform import image_trafo
from fit_lane import init_fit_line, similar_fit_line, meas_curv, draw_lane
import pickle
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

# threshold definition
thresh_r = (110, 255)
thresh_g = (100, 255)
thresh_s = (140, 240)
thresh_sobel = (25, 240)

# left lane
llane = Line()
# right lane
rlane = Line()


def img_pipeline(img):
    # undistort and transform image to birds-eye-view
    img, _, _, _, _, _, _, undist, M = image_trafo(img, mtx, dist, thresh_r, thresh_g, thresh_s, thresh_sobel,
                                                   undistort=1, perspective_transform=1)
    # first iteration calls the initial lane finding function
    if not llane.detected:
        init_fit_line(img, llane, rlane)
    else:
        # if the lane has been detected the previous run, do the simplified search
        similar_fit_line(img, llane, rlane)

    # measure the curvature of the lanes
    car_pos = meas_curv(llane, rlane)

    # draw the lines in the image and the value for the curvature
    img = draw_lane(img, undist, llane, rlane, M, car_pos)
    return img


white_output = 'project_video_output.mp4'
clip1 = VideoFileClip('project_video.mp4', audio=False)
white_clip = clip1.fl_image(img_pipeline)
white_clip.write_videofile(white_output, audio=False)
