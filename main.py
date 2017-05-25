from camera_calibration import calibrate_camera, undistort_images

# camera calibration
# get camera calibration values out of function by passing folder containing images, nx, ny and if you want a LOG
# and the result to be saved (will be saved as pickle file, first mtx, then dist)
ret, mtx, dist = calibrate_camera(folder='.\camera_cal/', nx=9, ny=6, save_result=0, save_log=0)


# create examples of undistorted images
undistort_images('.\camera_cal/', mtx, dist, nx=9, ny=6, chessboarddrawn=1)


# TODO: fiddle on test images
