import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

def canny(inp):

    gray_image = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5,5), 0)
    can = cv2.Canny(blur, 50, 150)

    return can

def roi(inp, region_of_interest):

    #channel = inp.shape[2]
    mask = np.zeros_like(inp)
    mask_color = 255
    cv2.fillPoly(mask, region_of_interest, mask_color)
    masked_image = cv2.bitwise_and(inp, mask)

    return masked_image

def coordinates(inp, values):

    slope, intercept = values

    y1 = inp.shape[0]
    y2 = int(y1 - 150)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

def hough_lines(inp, lines):

    left = []
    right = []

    for line in lines:
        #for x1, x2, y1, y2 in line:
            x1, y1, x2, y2 = line.reshape(4)
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if slope < 0:
                left.append((slope, intercept))
            else:
                right.append((slope, intercept))

    l_avg = np.average(left, axis=0)
    r_avg = np.average(right, axis=0)

    l_line = coordinates(inp, l_avg)
    r_line = coordinates(inp, r_avg)

    return np.array([l_line, r_line])

def visualize(inp, lines):

    lines_blank = np.zeros_like(inp).astype(np.uint8)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lines_blank, (x1, y1), (x2, y2), (0, 127, 127), 5)
            #cv2.fillPoly(lines_blank, np.array([[(x1, y1), (x2, y2)]]), (255,0,0) )
    return lines_blank

# read = img.imread('color.jpg')
# # #
# #region_of_interest = np.array([[(0, read.shape[0]), (read.shape[1]/2, read.shape[0]/1.7), (read.shape[1], read.shape[0])]])
# can = canny(read)
# region_of_interest = np.array([[(0, read.shape[0]), (800, read.shape[0]), (380, 290)]])
# cropped_region = roi(can, region_of_interest)
# hough = cv2.HoughLinesP(cropped_region, rho=2, theta=np.pi / 180, threshold=100, lines=np.array([]),
#                         minLineLength=50, maxLineGap=50)
# hough_transform = hough_lines(read, hough)
# lines = visualize(read, hough_transform)
# out = cv2.addWeighted(read, 0.9, lines, 1, 1)
# plt.imshow(out)
# plt.show()
#

capture = cv2.VideoCapture("input.mp4")

while (capture.isOpened()):

    get, read = capture.read()

    if get == True:

        region_of_interest = [(0, read.shape[0]), (800, read.shape[0]), (380, 290)]

        can = canny(read)
        cropped_region = roi(can, np.array([region_of_interest], np.int32))
        hough = cv2.HoughLinesP(cropped_region, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                                minLineLength=100, maxLineGap=50)
        hough_transform = hough_lines(read, hough)
        lines = visualize(read, hough_transform)
        out = cv2.addWeighted(read, 0.9, lines, 1, 1)

        cv2.imshow('frame', out)

        if cv2.waitKey(10) & 0xFF == ord('e'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
#
#
