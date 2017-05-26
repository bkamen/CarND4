import numpy as np
import cv2
import glob


def image_trafo_folder(folder, mtx, dist, thresh_r=(10, 255), thresh_h=(10, 255), thresh_s=(10, 255),
                       thresh_sobel=(10, 255), test_saves=0,  undistort=1, perspective_transform=1):
    # list of images that should be processed
    im_list = glob.glob(folder+'*.jpg')

    for i in im_list:
        img = cv2.imread(i)
        img = image_trafo(img, mtx, dist, thresh_r, thresh_h, thresh_s, thresh_sobel, test_saves, undistort,
                          perspective_transform)
        # define the filename for the output
        filename = './output_images/transformed_'+i.split('\\')[-1]
        # create the file
        cv2.imwrite(filename, img*255)


def image_trafo(img, mtx, dist, thresh_r=(10, 255), thresh_g=(10,255), thresh_h=(10, 255), thresh_s=(10, 255),
                thresh_sobel=(10, 255), test_saves=0,  undistort=1, perspective_transform=1):
    # function that returns a binary image which is a combination of the R channel of the BGR image, ...
    # the H and S channels of the HLS color space and color gradient in x direction (sobel x)
    # if user sets undistort flag, images will be undistorted
    if undistort:
        dst = cv2.undistort(img, mtx, dist, None, mtx)
    else:
        dst = img

    # calculate the R channel
    r = dst[:, :, 2]
    g = dst[:, :, 1]
    # calculate R binary
    r_binary = np.zeros_like(r)
    g_binary = np.zeros_like(g)
    r_binary[(r >= thresh_r[0]) & (r <= thresh_r[1])] = 1
    g_binary[(g >= thresh_g[0]) & (g <= thresh_g[1])] = 1

    # calculate the binaries of the H and S channel of the HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h = hls[:, :, 0]
    s = hls[:, :, 2]
    h_binary = np.ones_like(h)
    s_binary = np.zeros_like(s)
    h_binary[(h >= thresh_h[0]) & (h <= thresh_h[1])] = 0
    s_binary[(s >= thresh_s[0]) & (s <= thresh_s[1])] = 1

    # calculate the color gradient in x direction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    solx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # calculate color gradient with Sobel function of opencv
    abs_solx = np.abs(solx) # take the absolute
    norm_solx = np.uint8(255 * abs_solx / np.max(abs_solx))
    # calculate the binary of the normalized color gradient channel
    solx_binary = np.zeros_like(norm_solx)
    solx_binary[(norm_solx >= thresh_sobel[0]) & (norm_solx <= thresh_sobel[1])] = 1

    # combine all the binaries
    combined_binary = np.zeros_like(solx_binary)
    combined_binary[(r_binary == 1) | (h_binary == 1) | (s_binary == 1) | (solx_binary == 1)] = 1

    if test_saves:
        cv2.imwrite('test_r_channel.jpg', r_binary*255)
        cv2.imwrite('test_g_channel.jpg', g_binary*255)
        cv2.imwrite('test_h_channel.jpg', h_binary*255)
        cv2.imwrite('test_s_channel.jpg', s_binary*255)
        cv2.imwrite('test_solx_channel.jpg', solx_binary*255)

    # if the user specified a perspective transform, do it
    if perspective_transform:
        a=1

    return combined_binary
