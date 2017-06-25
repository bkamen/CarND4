import numpy as np
import cv2
import glob


def image_trafo_folder(folder, mtx, dist, thresh_r=(10, 255), thresh_g=(10,255), thresh_s=(10, 255),
                       thresh_sobel=(10, 255), test_saves=0,  undistort=1, perspective_transform=1):
    # list of images that should be processed
    im_list = glob.glob(folder+'*.jpg')

    for i in im_list:
        img = cv2.imread(i)
        img, r_binary, g_binary, rg_binary, \
        s_binary, v_binary, solx_binary, dst, M = image_trafo(img, mtx, dist, thresh_r, thresh_g, thresh_s,
                                                              thresh_sobel, undistort, perspective_transform)
        # define the filename for the output
        filename = './output_images/transformed_'+i.split('\\')[-1]
        # create the file
        cv2.imwrite(filename, cv2.resize(img*255, (0, 0), fx=.3, fy=.3))

        # if specified save the different channels of the test images seperately
        if test_saves:
            cv2.imwrite('./output_images/r_binary_'+i.split('\\')[-1], cv2.resize(r_binary * 255, (0, 0), fx=.3, fy=.3))
            cv2.imwrite('./output_images/g_binary_'+i.split('\\')[-1], cv2.resize(g_binary * 255, (0, 0), fx=.3, fy=.3))
            cv2.imwrite('./output_images/s_binary_'+i.split('\\')[-1], cv2.resize(s_binary * 255, (0, 0), fx=.3, fy=.3))
            cv2.imwrite('./output_images/rg_binary_'+i.split('\\')[-1], cv2.resize(rg_binary * 255, (0, 0), fx=.3, fy=.3))
            cv2.imwrite('./output_images/solx_binary_'+i.split('\\')[-1], cv2.resize(solx_binary * 255, (0, 0), fx=.3, fy=.3))
            cv2.imwrite('./output_images/v_binary_'+i.split('\\')[-1], cv2.resize(v_binary * 255, (0, 0), fx=.3, fy=.3))


def image_trafo(img, mtx, dist, thresh_r, thresh_g, thresh_s,
                thresh_sobel,  undistort=1, perspective_transform=1):
    # function that returns a binary image which is a combination of the R channel of the BGR image, ...
    # the H and S channels of the HLS color space and color gradient in x direction (sobel x)
    # if user sets undistort flag, images will be undistorted
    if undistort:
        dst = cv2.undistort(img, mtx, dist, None, mtx)
    else:
        dst = img

    # adjust the gamma level, so the yellow and white parts of the image stand out more
    gam = adjust_gamma(dst, .3)

    # calculate the R channel
    r = gam[:, :, 2]
    g = gam[:, :, 1]
    # calculate R and G binary
    r_binary = np.zeros_like(r)
    g_binary = np.zeros_like(g)
    rg_binary = np.zeros_like(r)
    r_binary[(r >= thresh_r[0]) & (r <= thresh_r[1])] = 1
    g_binary[(g >= thresh_g[0]) & (g <= thresh_g[1])] = 1
    # calculate the combined R and G binary, only true if R and G are in the limits
    rg_binary[(r_binary == 1) & (g_binary == 1)] = 1
    # calculate the binary of the S channel of the HLS color space
    hls = cv2.cvtColor(gam, cv2.COLOR_BGR2HLS)
    s = hls[:, :, 2]
    s_binary = np.zeros_like(s)
    s_binary[(s >= thresh_s[0]) & (s <= thresh_s[1])] = 1

    # calculate the binary of the V channel of the HVS color space
    hsv = cv2.cvtColor(gam, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_binary = np.zeros_like(v)
    v_binary[(v >= 220) & (v <= 255)] = 1

    # calculate the color gradient in x direction
    gray = cv2.cvtColor(gam, cv2.COLOR_BGR2GRAY)
    solx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # calculate color gradient with Sobel function of opencv
    abs_solx = np.abs(solx)  # take the absolute
    norm_solx = np.uint8(255 * abs_solx / np.max(abs_solx))
    # calculate the binary of the normalized color gradient channel
    solx_binary = np.zeros_like(norm_solx)
    solx_binary[(norm_solx >= thresh_sobel[0]) & (norm_solx <= thresh_sobel[1])] = 1

    # combine all the binaries
    combined_binary = np.zeros_like(solx_binary)
    combined_binary[(rg_binary == 1) | (s_binary == 1) | (v_binary == 1) |(solx_binary == 1)] = 1

    # if the user specified a perspective transform, do it
    if perspective_transform:
        warped, M = persp_transform(combined_binary)
    else:
        warped = combined_binary
        M = 0

    return warped, r_binary, g_binary, rg_binary, s_binary, v_binary, solx_binary, dst, M


def persp_transform(img):
    # function that performs perspective transform
    # source and destination are hardcoded and are found empirically by testing and deciding subjectively
    imshape = (img.shape[1], img.shape[0])
    offset = -30
    dst = np.float32([(475+offset, 100), (200+offset, imshape[1]), (imshape[0]-475+offset, 100), (imshape[0]-200+offset, imshape[1])])
    src = np.float32([(610, 470), (180, imshape[1]), (imshape[0] - 610, 470), (imshape[0] - 180, imshape[1])])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, imshape), M


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)