import numpy as np
import cv2

frame_number = 1
# -----------------------------------------------------------------------
# -------------------------FUNCTIONS DEFINITION--------------------------
# -----------------------------------------------------------------------


def region_of_interest(image, mask_vertices):
    """
     Only keeps the part of the image enclosed in the polygon and
     sets rest of the image to black
    """
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. Third element which will be 3 or 4
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, mask_vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    final_masked_image = cv2.bitwise_and(image, mask)

    return final_masked_image


def warp_image(image, mask_vertices):
    """
     This function takes in image and mask vertices as input, warps the image inside the mask
     to the width and height of the mask.
    """
    warped_image_height = abs(mask_vertices[0][0][1] - mask_vertices[0][3][1])
    warped_image_width = abs(mask_vertices[0][0][0] - mask_vertices[0][1][0])

    warped_img_vertices = np.float32([[0, warped_image_height],
                                      [warped_image_width, warped_image_height],
                                      [warped_image_width, 0],
                                      [0, 0]], dtype=np.int32)

    M = cv2.getPerspectiveTransform(np.float32(mask_vertices[0]), warped_img_vertices)
    final_warped_image = cv2.warpPerspective(image, M, (warped_image_width, warped_image_height))

    return final_warped_image, M


def compute_mask_vertices(img):
    """
     This function takes an image as input, requires the parameters to be set manually
     and generates the coordinates for the mask vertices.
    """
    # Region-of-interest vertices
    # For advanced.mp4 parameters are 0.6, 0.15, 0.25, 0.28, 0.12
    # For ottawa_video.mp4 parameters are 0.9, 0.25, 0.30, 0.05, 0.0
    # For nevada_video.mp4 parameters are 0.6, 0.25, 0.15, 0.0, 0.05
    bottom_width = 0.6  # width of bottom edge of trapezoid, expressed as percentage of image width
    top_width = 0.15  # ditto for top edge of trapezoid
    height = 0.25  # height of the trapezoid expressed as percentage of image height
    height_from_bottom = 0.28
    x_translation = 0.12  # Can be +ve or -ve. Translation of midpoint of region of interest along x axis

    img_shape = img.shape
    vertices = np.array(
        [[[(img_shape[1] * (1 - bottom_width)) // 2, int(img_shape[0] - img_shape[0] * height_from_bottom)],
          [int(img_shape[1] * bottom_width) + (img_shape[1] * (1 - bottom_width)) // 2,
           int(img_shape[0] - img_shape[0] * height_from_bottom)],
          [int(img_shape[1] * top_width) + (img_shape[1] * (1 - top_width)) // 2,
           int(img_shape[0] - img_shape[0] * height_from_bottom) - int(img_shape[0] * height)],
          [(img_shape[1] * (1 - top_width)) // 2,
           int(img_shape[0] - img_shape[0] * height_from_bottom) - int(img_shape[0] * height)]]],
        dtype=np.int32)

    vertices = np.array(vertices[:] - [x_translation * img_shape[1], 0], dtype='int')
    return vertices


def hsv_segmentation(image):
    """
    This function takes in an image, converts it to HSV color space, performs yellow color segmentation
    and returns a binary image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_bgr = np.uint8([[[0, 255, 255]]])  # yellow color in BGR
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)  # yellow color in HSV
    # print('-'*50, "\nHSV Thresholding")
    # print("Color in BGR = ", color_bgr)
    # print("Color in HSV = ", color_hsv)

    # Computing Threshold
    lower_threshold = np.array([color_hsv[0][0][0] - 10, 50, 50])
    upper_threshold = np.array([color_hsv[0][0][0] + 10, 200, 200])
    # print("Color lower threshold = ", lower_threshold, "Upper threshold = ", upper_threshold)
    # print('-'*50)

    # Computing pixels in hsv image that lie within the threshold limits
    binary_result_image = cv2.inRange(hsv, lower_threshold, upper_threshold)

    return binary_result_image


def sobel_thresholding(image):
    """
     This function uses absolute_sobel_thresholding(), sobel_gradMag_thresholding(),
     sobel_gradDir_thresholding() functions to threshold the input image, combines the results
     obtained from the three functions and returns a binary image.
    """

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # PARAMETER EXPERIMENTATION NOTE -
    # On OpenCV website it is recommended to use Scharr filter to improve results. To use Scharr filter
    # we pass (ksize = -1) as parameter. Scharr filter is said to give better results than 3x3 Sobel filter
    # Scharr filter(G_x) = [-3,0,3; -10,0,10; -3,0,3];  Sobel filter (G_x) = [-1,0,1; -2,0,2; -1,0,1]
    # A 3x3 Sobel operator may produce inaccuracies since it's an approximation of derivative
    #
    # NOTE: For lane detection a 3x3 Sobel filter produces better results than Scharr filter

    # Computing absolute sobel thresholded image
    sobel_absolute_binary = absolute_sobel_thresholding(image, threshold=(30, 100))
    # cv2.namedWindow("Absolute Sobel Result", cv2.WINDOW_NORMAL)
    # cv2.imshow("Absolute Sobel Result", sobel_absolute_binary)

    # Computing sobel gradient magnitude thresholded image
    sobel_gradMag_binary =sobel_gradMag_thresholding(image, threshold=(70, 120))
    # cv2.namedWindow("Sobel Gradient Magnitude Result", cv2.WINDOW_NORMAL)
    # cv2.imshow("Sobel Gradient Magnitude Result", sobel_gradMag_binary)

    # Computing sobel gradient direction thresholded image
    sobel_gradDir_binary = sobel_gradDir_thresholding(image, threshold=(40*np.pi/180, 70*np.pi/180))
    # cv2.namedWindow("Sobel Gradient Direction Result", cv2.WINDOW_NORMAL)
    # cv2.imshow("Sobel Gradient Direction Result", sobel_gradDir_binary)

    # Combining all the results
    combined_sobel_binary = np.zeros_like(image)
    combined_sobel_binary[(sobel_absolute_binary == 255) |
                          ((sobel_gradMag_binary == 255) & (sobel_gradDir_binary == 255))] = 255

    return combined_sobel_binary


def absolute_sobel_thresholding(image, threshold=(30, 100)):
    """
    This function takes in an image, applies Sobel derivative using x kernel, performs thresholding
    and returns a binary image. Best results have been found with threshold values of 30 and 100
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    absolute_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    scaled_absolute_sobel = np.uint8(255 * absolute_sobel / np.max(absolute_sobel))
    binary_result_image = np.zeros_like(scaled_absolute_sobel)
    binary_result_image[(absolute_sobel >= threshold[0]) & (absolute_sobel <= threshold[1])] = 255

    return binary_result_image


def sobel_gradMag_thresholding(image, threshold=(30,100)):
    """
    This function takes in an image, computes sobel derivatives in X and Y direction and uses them
    to compute gradient magnitude, performs thresholding and returns a binary image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    binary_result_image = np.zeros_like(gradient_magnitude)
    binary_result_image[(scaled_gradient_magnitude >= threshold[0]) & (scaled_gradient_magnitude <= threshold[1])] = 255

    return binary_result_image


def sobel_gradDir_thresholding(image, threshold=(30*np.pi/180, 70*np.pi/2)):
    """
    This function takes in an image, computes sobel derivatives in X and Y direction and uses them
    to compute gradient direction, performs thresholding and returns a binary image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Using absolute since gradient can be both positive and negative and we only care about magnitude
    absolute_gradient_direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_result_image = np.zeros_like(absolute_gradient_direction)
    binary_result_image[
        (absolute_gradient_direction >= threshold[0]) & (absolute_gradient_direction <= threshold[1])] = 255

    return binary_result_image


def lane_pixel_segmentation(image):
    """
    This function takes in a binary image obtained from thresholding to separate lane pixels,
    uses sliding window method to segment pixels into left side and right side line pixel that form
    the lane and returns four arrays with the pixel coordinates.
    """
    # Computing number of white pixels(value = 255) in each column. Basically a histogram
    histogram = np.sum(image, axis=0)
    histogram_midpoint = np.int(np.shape(histogram)[0] / 2)
    bottom_ten_percent = np.sum(image[np.int((1 - 0.1) * image.shape[0]):, :], axis=0)

    # Sliding window parameters
    number_of_windows = 9
    window_height = np.int(image.shape[0] / number_of_windows)

    # Identifying x and y position of all non zero pixels
    non_zero_pixels = image.nonzero()
    non_zero_y = np.array(non_zero_pixels[0])
    non_zero_x = np.array(non_zero_pixels[1])

    # Computing X and y coordinates of midpoint of the current window and setting them to the
    # coordinates of midpoint for the bottom most windows on left and right side
    current_window_leftx = np.argmax(bottom_ten_percent[0:histogram_midpoint])
    current_window_rightx = np.argmax(bottom_ten_percent[histogram_midpoint:-1]) + histogram_midpoint

    # Width of the windows. Here width refers to size of window from the midpoint of the window
    window_width = np.int(image.shape[1] * 0.07)

    # Initializing arrays to store the indices of pixels on left and right side
    left_lane_pixel_indices = []
    right_lane_pixel_indices = []

    # Initializing an image to display output of this function
    window_output_image = np.dstack((image, image, image)).astype('uint8')

    # Stepping through windows one by one. Building them from bottom up
    for window_number in range(number_of_windows):
        # Computing vertices of left and right window
        window_bottom_y = image.shape[0] - window_number * window_height
        window_top_y = image.shape[0] - (window_number + 1) * window_height
        left_window_leftx = current_window_leftx - window_width
        left_window_rightx = current_window_leftx + window_width
        right_window_leftx = current_window_rightx - window_width
        right_window_rightx = current_window_rightx + window_width

        # Drawing windows in the output image
        cv2.rectangle(window_output_image, (left_window_leftx, window_bottom_y),
                      (left_window_rightx, window_top_y), (0, 255, 0), 2)
        cv2.rectangle(window_output_image, (right_window_leftx, window_bottom_y),
                      (right_window_rightx, window_top_y), (0, 255, 0), 2)

        # Identifying non-zero pixels which are inside to left and right window
        left_window_pixels_array_location = ((non_zero_y >= window_top_y) &
                                             (non_zero_y <= window_bottom_y) &
                                             (non_zero_x >= left_window_leftx) &
                                             (non_zero_x <= left_window_rightx)).nonzero()[0]
        right_window_pixels_array_location = ((non_zero_y >= window_top_y) &
                                              (non_zero_y <= window_bottom_y) &
                                              (non_zero_x >= right_window_leftx) &
                                              (non_zero_x <= right_window_rightx)).nonzero()[0]

        # Appending
        left_lane_pixel_indices.append(left_window_pixels_array_location)
        right_lane_pixel_indices.append(right_window_pixels_array_location)

        # Recentering the next window on the mean position of the pixels of the current window
        # If condition checks for if there are pixels in the window
        top_five_percent_left = ((non_zero_y >= window_top_y) &
                                 (non_zero_y <= window_top_y + 0.05 * window_height) &
                                 (non_zero_x >= left_window_leftx) &
                                 (non_zero_x <= left_window_rightx)).nonzero()[0]
        if top_five_percent_left.size:
            # current_window_leftx = np.int(np.median(non_zero_x[left_window_pixels_array_location]))
            current_window_leftx = np.int(np.median(non_zero_x[top_five_percent_left]))

        top_five_percent_right = ((non_zero_y >= window_top_y) &
                                  (non_zero_y <= window_top_y + 0.05 * window_height) &
                                  (non_zero_x >= right_window_leftx) &
                                  (non_zero_x <= right_window_rightx)).nonzero()[0]
        if top_five_percent_right.size:
            # current_window_rightx = np.int(np.median(non_zero_x[right_window_pixels_array_location]))
            current_window_rightx = np.int(np.median(non_zero_x[top_five_percent_right]))

        # Checking if the previous windows were at the correct position
        if window_number > 0:
            previous_window_number = window_number - 1
            previous_window_leftx = current_window_leftx
            previous_window_rightx = current_window_rightx
            while previous_window_number >= 1:
                previous_window_bottom_y = image.shape[0] - (previous_window_number) * window_height
                previous_window_top_y = window_bottom_y
                previous_left_window_leftx = previous_window_leftx - window_width
                previous_left_window_rightx = previous_window_leftx + window_width
                previous_right_window_leftx = previous_window_rightx - window_width
                previous_right_window_rightx = previous_window_rightx + window_width

                previous_left_window_pixels_array_location = ((non_zero_y >= previous_window_top_y) &
                                                     (non_zero_y <= previous_window_bottom_y) &
                                                     (non_zero_x >= previous_left_window_leftx) &
                                                     (non_zero_x <= previous_left_window_rightx)).nonzero()[0]

                previous_right_window_pixels_array_location = ((non_zero_y >= previous_window_top_y) &
                                                      (non_zero_y <= previous_window_bottom_y) &
                                                      (non_zero_x >= previous_right_window_leftx) &
                                                      (non_zero_x <= previous_right_window_rightx)).nonzero()[0]

                change_in_left = (np.shape(previous_left_window_pixels_array_location)[0] - np.shape(left_lane_pixel_indices[previous_window_number])[0])
                if change_in_left > 20:
                    left_lane_pixel_indices[previous_window_number] = np.append(left_lane_pixel_indices[previous_window_number], previous_left_window_pixels_array_location)
                    previous_window_leftx = np.int(np.median(non_zero_x[previous_left_window_pixels_array_location]))
                change_in_right = (np.shape(previous_right_window_pixels_array_location)[0] - np.shape(right_lane_pixel_indices[previous_window_number])[0])
                if change_in_right > 20:
                    right_lane_pixel_indices[previous_window_number] = np.append(right_lane_pixel_indices[previous_window_number], previous_right_window_pixels_array_location)
                    previous_window_rightx = np.int(np.median(non_zero_x[previous_right_window_pixels_array_location]))

                if change_in_left <= 20 and change_in_right <= 20:
                    break
                previous_window_number = previous_window_number - 1

    # Concatenating arrays because after appending in each loop
    # the left_lane_pixel_indices and right_lane_pixel_indices
    # are in form of nested lists or a 2-d array
    left_lane_pixel_indices = np.concatenate(left_lane_pixel_indices)
    right_lane_pixel_indices = np.concatenate(right_lane_pixel_indices)

    # Extracting the x and y coordinates of pixels for left and right lanes
    left_lane_x = non_zero_x[left_lane_pixel_indices]
    left_lane_y = non_zero_y[left_lane_pixel_indices]
    right_lane_x = non_zero_x[right_lane_pixel_indices]
    right_lane_y = non_zero_y[right_lane_pixel_indices]

    # In output image changing color of pixels in left lane to blue
    # and pixels in right lane to red
    window_output_image[non_zero_y[left_lane_pixel_indices], non_zero_x[left_lane_pixel_indices]] = [255, 0, 0]
    window_output_image[non_zero_y[right_lane_pixel_indices], non_zero_x[right_lane_pixel_indices]] = [0, 0, 255]

    # Displaying results of Sliding Window and Segmenting pixels into left and right lane
    cv2.namedWindow("Sliding Window and Lane Pixel Segmentation Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Sliding Window and Lane Pixel Segmentation Result", window_output_image)

    return left_lane_x, left_lane_y, right_lane_x, right_lane_y


def lane_line_polyfit(y_coordinates, lane_pixels_y, lane_pixels_x):
    """
     This function takes in the x and y coordinates of the lane pixels, fit a 2 degree polynomial curve
     through them, computer the x and y coordinates that lie on the curve and returns an array of coordinates
    """
    lane_line_parameters = np.polyfit(lane_pixels_y, lane_pixels_x, 2)
    lane_line_x_coordinates = lane_line_parameters[0]*y_coordinates**2 + \
                              lane_line_parameters[1]*y_coordinates + \
                              lane_line_parameters[2]

    pts = np.array(np.transpose(np.vstack([lane_line_x_coordinates, y_coordinates])), np.int)
    pts = pts.reshape(-1, 1, 2)

    return pts


def fit_lane_lines(image, left_lane_x, left_lane_y, right_lane_x, right_lane_y):
    """
    This function takes in the warped image to compute y-coordinates and also the coordinates of
    lane line pixels, computes 2-d polynomial that fit the pixel coordinates and returns
    two array. One for left and one for right side polynomial line coordinate that fit the pixels
    """
    y_coordinates = np.linspace(0, image.shape[0] - 1, image.shape[0])
    if len(right_lane_x) == 0 or len(right_lane_x) == 0:
        pts_left = lane_line_polyfit(y_coordinates, left_lane_y, left_lane_x)
        pts_right = []
        return pts_left, pts_right
    elif len(left_lane_x) == 0 or len(left_lane_x) == 0:
        pts_left = []
        pts_right = lane_line_polyfit(y_coordinates, right_lane_y, right_lane_x)
        return pts_left, pts_right
    if (len(right_lane_x) == 0 or len(right_lane_x) == 0) and (len(left_lane_x) == 0 or len(left_lane_x) == 0):
        pts_left = []
        pts_right = []
        return pts_left, pts_right
    else:
        pts_left = lane_line_polyfit(y_coordinates, left_lane_y, left_lane_x)
        pts_right = lane_line_polyfit(y_coordinates, right_lane_y, right_lane_x)
        return pts_left, pts_right


def draw_lane_lines(original_image, binary_warped_image, left_pts, right_pts):
    """
    This function computes the final result image
    """
    lane_lines = np.zeros((binary_warped_image.shape[0], binary_warped_image.shape[1], 3), dtype='uint8')
    # Drawing the lane lines on the colored warped image
    if len(left_pts) == 0:
        cv2.polylines(lane_lines, [right_pts], isClosed=False, color=(0, 0, 255), thickness=15)
    elif len(right_pts) == 0:
        cv2.polylines(lane_lines, [left_pts], isClosed=False, color=(0, 0, 255), thickness=15)
    elif len(left_pts) == 0 and len(right_pts) == 0:
        pass
    else:
        cv2.polylines(lane_lines, [left_pts], isClosed=False, color=(0, 0, 255), thickness=15)
        cv2.polylines(lane_lines, [right_pts], isClosed=False, color=(0, 0, 255), thickness=15)

    # Warping warped image back to original image
    warped_back_image = cv2.warpPerspective(lane_lines, np.linalg.inv(M), (img.shape[1], img.shape[0]))

    # Finding all the lane pixels in the warped back image which are non-zero in the red channel
    lane_pixels = (warped_back_image[:, :, 2]).nonzero()

    # Making the identified lane pixels in the original image as red
    original_image[np.array(lane_pixels[0]), np.array(lane_pixels[1])] = (0, 0, 255)

    return original_image


# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

if __name__ == '__main__':
    cap = cv2.VideoCapture('advanced.mp4')

    # Defining codec and creating video writer object
    # **NOTE** - Make sure to use the correct codec and video format combination
    # Check for codecs installed on system
    # Also make sure that the size of the frame is correct otherwise the output video file won't play
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('advanced_result.avi', fourcc, 25.0, (1280, 720))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        print("Frame number = ", frame_number)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Computing vertices and displaying mask portion of image
        vertices = compute_mask_vertices(img)
        masked_image = region_of_interest(img, vertices)
        # cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Masked Image", masked_image)

        # Extracting yellow lane line from warped image using H Channel Color Segmentation from HSV
        binary_color_segmentation = hsv_segmentation(img)
        # cv2.namedWindow("Color Segmentation Result", cv2.WINDOW_NORMAL)
        # cv2.imshow("Color Segmentation Result", binary_color_segmentation)

        # Extracting white and yellow lane lines using Sobel operator
        binary_sobel_thresholding = sobel_thresholding(img)
        # cv2.namedWindow("Sobel Thresholding Result", cv2.WINDOW_NORMAL)
        # cv2.imshow("Sobel Thresholding Result", binary_sobel_thresholding)

        # Combining results of HSV and Sobel Thresholding
        binary_combined = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        binary_combined[(binary_sobel_thresholding == 255) | (binary_color_segmentation == 255)] = 255
        # cv2.namedWindow("Combined Result of HSV and Sobel Thresholding", cv2.WINDOW_NORMAL)
        # cv2.imshow("Combined Result of HSV and Sobel Thresholding", binary_combined)

        warped_image, M = warp_image(binary_combined, vertices)
        # cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Warped Image", warped_image)

        # Closing the warped image
        warped_image = cv2.dilate(warped_image, np.ones((7, 7), np.uint8), iterations=1)
        warped_image = cv2.erode(warped_image, np.ones((3,3), np.uint8), iterations=1)

        # Performing lane pixels segmentation and dividing them into x and y coordinates
        # of left and right side line of the highway lane
        left_lane_x, left_lane_y, right_lane_x, right_lane_y = lane_pixel_segmentation(warped_image)

        # Computing left and right side lines
        left_line_pts, right_line_pts = fit_lane_lines(warped_image, left_lane_x,
                                                      left_lane_y, right_lane_x, right_lane_y)

        # Visualizing final result
        final_result_image = draw_lane_lines(img, warped_image, left_line_pts, right_line_pts)
        cv2.namedWindow("Final Result Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Result Image", final_result_image)

        frame_number += 1
        out.write(final_result_image)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

out.release()
cap.release()
cv2.destroyAllWindows()

