import numpy as np
import cv2

# -----------VARIABLES---------------
# Parameters for region of interest
# For highway.mp4 parameters are 1, 0.2, 0.3, 0
# For highway_long.mp4 parameters are 0.8, 0.1, 0.3, 0
# For night.mp4 parameters are 0.8, 0.4, 0.3, 0
bottom_width = 0.8  # width of bottom edge of trapezoid, expressed as percentage of image width
top_width = 0.1  # ditto for top edge of trapezoid
height = 0.3  # height of the trapezoid expressed as percentage of image height
height_from_bottom = 0 # height from bottom as percentage of image height
# -----------------------------------------


def canny_edge_median(img):
    """canny_edge_median takes an image and does auto-thresholding
    using median to compute the edges using canny edge technique
    """
    median = np.median(img)
    low_threshold = median * 0.66
    upper_threshold = median * 1.33
    return cv2.Canny(img, low_threshold, upper_threshold)


def canny_edge_mean(img):
    """canny_edge_mean takes an image and does auto-thresholding
    using mean to compute the edges using canny edge technique
    """
    mean = np.mean(img)
    low_threshold = mean * 0.66
    upper_threshold = mean * 1.33
    return cv2.Canny(img, low_threshold, upper_threshold)


def region_of_interest(img, vertices):
    """
     Only keeps the part of the image enclosed in the polygon and
     sets rest of the image to black
    """
    mask = np.zeros_like(img)
    
    mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def hough_transform(img):
    """
     Computes lines using the probabilistic hough transform provided by OpenCV
     Thus it computes lines of finite size and returns them in form of an array

    :param img: masked edge detected image with only region of interest
    :return:
    """
    # Parameters
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    return lines


def filter_horizontal_lines(lines):
    """
     Removes all lines with slope between -10 and +10 degrees
     This is done because for highway lane lines the lines will be closer to being
     vertical from the view of the front mounted camera
    """
    non_horizontal_lines = [l for l in lines if
                            not -10 <= np.rad2deg(np.arctan((l[0][3] - l[0][1]) / (l[0][2] - l[0][0]))) <= 10]
    non_horizontal_lines = np.array(non_horizontal_lines)
    return non_horizontal_lines


def separate_lines(lines):
    """
    Separates the left and right lines of the highway lane
    :param lines: an array containing the lines which make left and right side of highway lane
    """
    left_lines = []
    right_lines = []
    # Here we separate coordinates of left and right-side lines of the highway lane
    # Since the y-axis is positive in downwards direction and x-axis is positive in right hand direction
    # With origin at the top left corner of the image
    # A negative slope will mean that the line is on the left ( in normal coordinate system it
    # will mean on the right side)
    # A positive slope will mean that the line is on the right ( in normal coordinate system it
    # will mean on the left side)

    for l in lines:
        slope = (l[0][3] - l[0][1]) / (l[0][2] - l[0][0])
        if slope < 0:
            # Slope is negative hence line is on the left side
            left_lines += [(l[0][0], l[0][1], l[0][2], l[0][3])]
        elif slope > 0:
            # Slope is positive hence line is on the right side
            right_lines += [(l[0][0], l[0][1], l[0][2], l[0][3])]
        else:
            print("Something looks fishy here")

    return left_lines, right_lines


def filter_lane_lines(left_lines, right_lines):
    """
    This function removes lines from left_lines that are closer to the right-side of the highway lane
    and from right_lines removes lines that are closer to left-side of highway lane. It also removes
    the lines which are more or less than 10 degrees from the median slope of each side.
    """
    if len(left_lines) == 0 or len(right_lines) == 0:
        return left_lines, right_lines

    # Filtering lines that lie close to the other side, for instance
    # lines in left_lines array that are closer to the right lane line
    x_top_left = []
    for x1, y1, x2, y2 in left_lines:
        x_top_left += [x2]
    x_top_left_median = np.median(x_top_left)
    left_lines_final = [l for l in left_lines if l[2] <= x_top_left_median]

    slope_left_lines = []
    for x1, y1, x2, y2 in left_lines_final:
        slope_left_lines += [np.rad2deg(np.arctan((y2 - y1) / (x2 - x1)))]

    x_top_right = []
    for x1, y1, x2, y2 in right_lines:
        x_top_right += [x1]
    x_top_right_median = np.median(x_top_right)
    right_lines_final = [l for l in right_lines if l[0] >= x_top_right_median]

    slope_right_lines = []
    for x1, y1, x2, y2 in right_lines_final:
        slope_right_lines += [np.rad2deg(np.arctan((y2 - y1)/(x2 - x1)))]

    # Filtering based on slope
    median_left_lines_slope = np.median(slope_left_lines)
    left_lines_final_filtered = []
    for i in range(len(left_lines_final)):
        if (-1 + median_left_lines_slope) <= slope_left_lines[i] <= (10 + median_left_lines_slope):
            left_lines_final_filtered += [left_lines_final[i]]

    median_right_lines_slope = np.median(slope_right_lines)
    right_lines_final_filtered = []
    for i in range(len(right_lines_final)):
        if (-5 + median_right_lines_slope) <= slope_right_lines[i] <= (5 + median_right_lines_slope):
            right_lines_final_filtered += [right_lines_final[i]]

    return left_lines_final_filtered, right_lines_final_filtered


def draw_single_line(lines):
    """
    Takes in an array of lines and combines them into a single line
    """
    if len(lines) == 0:
        return [0, 0, 0, 0]

    # Maximum and minimum y-coordinate for the sigle line on left and right side
    y_max = int(img.shape[0] - img.shape[0] * height_from_bottom)
    y_min = int(img_shape[0] - img_shape[0] * height_from_bottom) - int(img_shape[0] * height)

    # Computing the top and bottom x co-ordinate obtained by extrapolating
    # the limited length lines.
    x_top = []
    x_bottom = []
    for x1, y1, x2, y2 in lines:
        z = np.polyfit([x1, x2], [y1, y2], 1)
        m, c = z
        x_top.append(int((y_min - c) / m))
        x_bottom.append(int((y_max - c) / m))

    x_avg_top = np.int(np.median(x_top))
    x_avg_bottom = np.int(np.median(x_bottom))

    return [x_avg_bottom, y_max, x_avg_top, y_min]


def highway_lane_lines(img):
    """
    Computes hough transform, separates lines on left and right side of the highway lane computed
    by hough transform, then forms a single line on the right side and left side
    """

    # Computing lines with hough transform
    lines = hough_transform(img)
    # Removing horizontal lines detected from hough transform
    lane_lines = filter_horizontal_lines(lines)

    # Separating lines on left and right side of the highway lane
    left_lines, right_lines = separate_lines(lane_lines)

    # Filtering lines i.e. removing left lines that are closer to right side and vice versa
    left_lines, right_lines = filter_lane_lines(left_lines, right_lines)

    # Computing one single line for left and right side
    left_side_line = draw_single_line(left_lines)
    right_side_line = draw_single_line(right_lines)

    return left_side_line, right_side_line


if __name__ == '__main__':
    cap = cv2.VideoCapture('highway_long.mp4')

    # Defining codec and creating video writer object
    # **NOTE** - Make sure to use the correct codec and video format combination
    # Check for codecs installed on system
    # Also make sure that the size of the frame is correct otherwise the output video file won't play
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('night.avi', fourcc, 25.0, (720, 540))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Image size = ", img.shape)

        # Applying gaussian blur
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Computing edges
        img_edges = canny_edge_median(img)

        # Computing region of interest
        img_shape = img.shape
        my_vertices = np.array(
            [[[(img_shape[1] * (1 - bottom_width)) // 2, int(img_shape[0] - img_shape[0] * height_from_bottom)],
              [int(img_shape[1] * bottom_width) + (img_shape[1] * (1 - bottom_width)) // 2,
               int(img_shape[0] - img_shape[0] * height_from_bottom)],
              [int(img_shape[1] * top_width) + (img_shape[1] * (1 - top_width)) // 2,
               int(img_shape[0] - img_shape[0] * height_from_bottom) - int(img_shape[0] * height)],
              [(img_shape[1] * (1 - top_width)) // 2,
               int(img_shape[0] - img_shape[0] * height_from_bottom) - int(img_shape[0] * height)]]],
            dtype=np.int32)

        masked_image = region_of_interest(img_edges, my_vertices)

        # Computing lane lines
        final_left_line, final_right_line = highway_lane_lines(masked_image)

        cv2.line(frame, (final_left_line[0], final_left_line[1]), (final_left_line[2], final_left_line[3]), (0, 255, 0), 5)
        cv2.line(frame, (final_right_line[0], final_right_line[1]), (final_right_line[2], final_right_line[3]), (0, 0, 255), 5)

        # Write frame to output file
        out.write(frame)
        cv2.imshow("Final Video", frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
