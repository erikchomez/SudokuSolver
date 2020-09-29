import cv2
import numpy as np
from operator import itemgetter

DEBUG = True


def main():
    image = cv2.imread('sudoku.jpeg')

    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply Gaussian blur to grayscale image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if DEBUG:
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)

    # invert threshold
    thresh = cv2.bitwise_not(thresh)

    if DEBUG:
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)

    # finds the contours in the threshold image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sorts the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # initializes a place holder for the puzzle contour
    puzzle_contour = None

    for cnt in contours:
        # calculates the perimeter or arc length of a closed contour
        perimeter = cv2.arcLength(cnt, True)
        # approximates the contour shape
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        # if the contour has 4 vertices, then we found the grid
        if len(approx) == 4:
            puzzle_contour = cnt
            break
    # in the case that the outline is not found, we throw an exception
    if puzzle_contour is None:
        raise Exception('No grid outline found.')

    #if DEBUG:
        # draw the grid outline onto the original image
        #cv2.drawContours(image, [puzzle_contour], -1, (0, 255, 0), 2)
        #cv2.imshow('outline', image)
        #cv2.waitKey(0)

    """
    Bottom right point has the largest (x + y) value
    Top left point has the smallest (x + y) value
    Bottom left point has the smallest (x - y) value
    Top right point has the largest (x - y) value
    """

    # the following returns the index of each point
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in approx]))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in approx]))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in approx]))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in approx]))

    print(approx[bottom_right])
    print(approx[top_left])
    print(approx[bottom_left])
    print(approx[top_right])

    red = [0, 0, 255]

    print(tuple(approx[bottom_right][0]))
    cv2.circle(image, tuple(approx[bottom_right][0]), 1, red, 3)
    cv2.circle(image, tuple(approx[top_left][0]), 1, red, 3)
    cv2.circle(image, tuple(approx[bottom_left][0]), 1, red, 3)
    cv2.circle(image, tuple(approx[top_right][0]), 1, red, 3)

    if DEBUG:
        cv2.imshow('outline', image)
        cv2.waitKey(0)




if __name__ == '__main__':
    main()
