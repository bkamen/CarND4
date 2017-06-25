## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In the file 'camera_calibration.py' there is the function 'calibrate_camera'. A folder which should contain the images is passed to the function as well as the number of grids in x and y.
The user can specify if the result should be saved and if the logfile should be saved. The logfile contains a protocol of the process.

For each image the function 'corner_detection' is called first in which the image is converted to grayscale and after that the opencv function cv2.findChessboardCorners is called. It returns the corner matrix.

To transform the image an array of object points is needed (variable objp in 'camera_calibration.py', line 15). The array objp should represent the chessboard corner in the world frame which is assumed to be planar. Since this is the same for every image the array objp is appended to the array objpts for every image.
The image points are the identified corners of the chessboards and are appended to the array imgpts for every image.

To get the calibration coefficients the opencv function cv2.calibrateCamera is called ('camera_calibration.py', line 40) which calculates the coeffs based on the objpts and imgpts array.
If defined by the user the coefficients will be saved as pickle file. Same goes for the logfile which is saved as textfile.

![Original image](./camera_cal/calibration1.jpg) ![undistorted image](./output_images/undistorted_calibration1.jpg)
![Original image](./camera_cal/calibration2.jpg) ![undistorted image](./output_images/undistorted_calibration2.jpg)
![Original image](./camera_cal/calibration3.jpg) ![undistorted image](./output_images/undistorted_calibration3.jpg)
![Original image](./camera_cal/calibration4.jpg) ![undistorted image](./output_images/undistorted_calibration4.jpg)
![Original image](./camera_cal/calibration5.jpg) ![undistorted image](./output_images/undistorted_calibration5.jpg)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Original and undistored example:
![Undistorted example](./output_images/undistorted_straight_lines1.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The method used consists of using single channels of 3 color spaces plus the sobel gradient calculation in x direction.
A combination of the used for the output image.

Of the BGR color space the R and G channels are used. Lane lines always have R components as well as G components (usually white or yellow lines).
To not run into the issue to also have a yellowish tarmac included an elementwise matrix multiplication of R and G channel is performed to only catch pixels that have both components ('image_transform.py', line 47-54).

R and G binaries:

![R binary](r_binary_straight_lines1.jpg) ![R binary](r_binary_straight_lines2.jpg) ![R binary](r_binary_test1.jpg) ![R binary](r_binary_test2.jpg) ![R binary](r_binary_test3.jpg) ![R binary](r_binary_test4.jpg) ![R binary](r_binary_test5.jpg) ![R binary](r_binary_test6.jpg)
![G binary](g_binary_straight_lines1.jpg) ![G binary](g_binary_straight_lines2.jpg) ![G binary](g_binary_test1.jpg) ![G binary](g_binary_test2.jpg) ![G binary](g_binary_test3.jpg) ![G binary](g_binary_test4.jpg) ![G binary](g_binary_test5.jpg) ![G binary](g_binary_test6.jpg)

Combined R and G binaries:
![RG binary](rg_binary_straight_lines1.jpg) ![RG binary](rg_binary_straight_lines2.jpg) ![RG binary](rg_binary_test1.jpg) ![RG binary](rg_binary_test2.jpg) ![RG binary](rg_binary_test3.jpg) ![RG binary](rg_binary_test4.jpg) ![RG binary](rg_binary_test5.jpg) ![RG binary](rg_binary_test6.jpg)


Of the HLS color space the S channel is used ('image_transform.py', line 55-59):
![S binary](s_binary_straight_lines1.jpg) ![S binary](s_binary_straight_lines2.jpg) ![S binary](s_binary_test1.jpg) ![S binary](s_binary_test2.jpg) ![S binary](s_binary_test3.jpg) ![S binary](s_binary_test4.jpg) ![S binary](s_binary_test5.jpg) ![S binary](s_binary_test6.jpg)

Of the HSV color space the V channel is used ('image_transform.py', line 61-65):
![V binary](v_binary_straight_lines1.jpg) ![V binary](v_binary_straight_lines2.jpg) ![V binary](v_binary_test1.jpg) ![V binary](v_binary_test2.jpg) ![V binary](v_binary_test3.jpg) ![V binary](v_binary_test4.jpg) ![V binary](v_binary_test5.jpg) ![V binary](v_binary_test6.jpg)

And last but not least a binary of the sobel gradient in the x-direction of a grayscale image is used ('image_transform.py', line 67-74):
![Sobelx binary](solx_binary_straight_lines1.jpg) ![Sobelx binary](solx_binary_straight_lines2.jpg) ![Sobelx binary](solx_binary_test1.jpg) ![Sobelx binary](solx_binary_test2.jpg) ![Sobelx binary](solx_binary_test3.jpg) ![Sobelx binary](solx_binary_test4.jpg) ![Sobelx binary](solx_binary_test5.jpg) ![Sobelx binary](solx_binary_test6.jpg)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
