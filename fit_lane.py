import numpy as np
import matplotlib.pyplot as plt
import cv2


def init_fit_line(img, llane, rlane):
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

    out_img = np.dstack((img, img, img))*255

    mid = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:mid])
    rightx_base = np.argmax(histogram[mid:]) + mid

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()

    llane.detected = True
    llane.current_fit = left_fit
    llane.best_fit = left_fit
    llane.recent_xfitted = leftx
    llane.allx = leftx
    llane.ally = lefty
    llane.fits = np.expand_dims(left_fit, axis=0)

    rlane.detected = True
    rlane.current_fit = right_fit
    rlane.best_fit = right_fit
    rlane.recent_xfitted = rightx
    rlane.allx = rightx
    rlane.ally = righty
    rlane.fits = np.expand_dims(right_fit, axis=0)


def similar_fit_line(img, llane, rlane):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "img")
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (llane.current_fit[0]*(nonzeroy**2) + llane.current_fit[1]*nonzeroy + llane.current_fit[2] - margin)) & (nonzerox < (llane.current_fit[0]*(nonzeroy**2) + llane.current_fit[1]*nonzeroy + llane.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (rlane.current_fit[0]*(nonzeroy**2) + rlane.current_fit[1]*nonzeroy + rlane.current_fit[2] - margin)) & (nonzerox < (rlane.current_fit[0]*(nonzeroy**2) + rlane.current_fit[1]*nonzeroy + rlane.current_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if 0.85 * llane.current_fit[0] <= left_fit[0] <= 1.15 * llane.current_fit[0] or\
        0.85 * llane.current_fit[1] <= left_fit[1] <= 1.15 * llane.current_fit[1] or \
            0.95 * llane.current_fit[2] <= left_fit[2] <= 1.05 * llane.current_fit[2]:
        llane.detected = True
        llane.current_fit = left_fit
        llane.recent_xfitted = left_fitx
        llane.allx = leftx
        llane.ally = lefty
        if np.shape(llane.fits)[0] > 5:
            llane.fits = np.append(llane.fits[1:], [left_fit], axis=0)
        else:
            llane.fits = np.append(llane.fits, [left_fit], axis=0)
        llane.best_fit = np.mean(llane.fits, 0)
    else:
        llane.current_fit = llane.best_fit

    if 0.85 * rlane.current_fit[0] <= right_fit[0] <= 1.15 * rlane.current_fit[0] or \
        0.85 * rlane.current_fit[1] <= right_fit[1] <= 1.15 * rlane.current_fit[1] or \
            0.95 * rlane.current_fit[2] <= right_fit[2] <= 1.05 * rlane.current_fit[2]:
        rlane.detected = True
        rlane.current_fit = right_fit
        rlane.recent_xfitted = right_fitx
        rlane.allx = rightx
        rlane.ally = righty
        if np.shape(rlane.fits)[0] > 5:
            rlane.fits = np.append(rlane.fits[1:], [right_fit], axis=0)
        else:
            rlane.fits = np.append(rlane.fits, [right_fit], axis=0)
        rlane.best_fit = np.mean(rlane.fits, 0)
    else:
        rlane.current_fit = rlane.best_fit


def meas_curv(llane, rlane):
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = llane.best_fit[0] * ploty ** 2 + llane.best_fit[1] * ploty + llane.best_fit[2]
    rightx = rlane.best_fit[0] * ploty ** 2 + rlane.best_fit[1] * ploty + rlane.best_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 40 / 520  # meters per pixel in y dimension
    xm_per_pix = 4 / 1080  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    llane.radius_of_curvature = left_curverad
    rlane.radius_of_curvature = right_curverad


def draw_lane(warped, undist, llane, rlane, M):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    leftx = llane.current_fit[0] * ploty ** 2 + llane.current_fit[1] * ploty + llane.current_fit[2]
    rightx = rlane.current_fit[0] * ploty ** 2 + rlane.current_fit[1] * ploty + rlane.current_fit[2]
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # calculate inverse transformation matrix
    Minv = np.linalg.inv(M)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # write the curvature of both lanes
    font = cv2.FONT_HERSHEY_SIMPLEX
    strleft = 'Radius of left lane [m]: ' + str(llane.radius_of_curvature)
    strright = 'Radius of right lane [m]: ' + str(rlane.radius_of_curvature)
    cv2.putText(result, strleft, (50, 100), font, 2, (255, 255, 255))
    cv2.putText(result, strright, (50, 200), font, 2, (255, 255, 255))
    return result
