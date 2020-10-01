import cv2
import numpy as np
import cnn

from operator import itemgetter

DEBUG = False
RED = [0, 0, 255]


def show_image(image, image_str):
    """
    Quick function that displays an image
    :param image: image to display
    :param image_str: image title
    :return: None
    """
    cv2.imshow(image_str, image)
    cv2.waitKey(0)


def process_image(image, apply=True):
    """
    Applies pre-processing to an image. This includes converting to grayscale, applying a GaussianBlur, Adaptive
    Threshold, and inverting the thresh using bitwise_not
    :param image: image to be processed
    :param apply: bool to determine whether or not to invert the threshold, placeholder for when we train the model
    :return: processed image
    """
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply Gaussian blur to grayscale image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if apply:
        # invert threshold
        thresh = cv2.bitwise_not(thresh)

    return thresh


def find_contours(threshed_image):
    """
    Finds the contours in an image that has been threshed
    :param threshed_image: the threshed image
    :return: numpy array of the found contours
    """
    # finds the contours in the threshold image
    contours, _ = cv2.findContours(threshed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sorts the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def find_corners(contours):
    """
    Finds the four corners of the Sudoku grid. Throws an exception if no grid is found.
    :param contours: contours to check
    :return: array of coordinates
    """
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

    return approx


def distance_between(p1, p2):
    """
    Computes and returns the scalar distance between two pairs of points
    :param p1: the first pair coordinates
    :param p2: the second pair coordinates
    :return: scalar distance
    """
    return np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[0]) ** 2))


def perspective_transform(orig_image, corners):
    """
    Warps and transforms the image.

    Top left point has the smallest (x + y) value,
    Top right point has the largest (x - y) value
    Bottom right point has the largest (x + y) value
    Bottom left point has the smallest (x - y) value
    :param orig_image: the original image to apply the transformation to
    :param corners: array of corner coordinates
    :return: the warped image
    """
    # creates a placeholder for the rectangle array
    rect = np.zeros((4, 2), dtype='float32')
    # computes the sums for each pair of coordinates in the grid
    sums = [np.sum(i) for i in corners]
    # computes the differences for each pair of coordinates in the grid
    diffs = [np.diff(j) for j in corners]

    rect[0] = corners[np.argmin(sums)]  # top left
    rect[1] = corners[np.argmin(diffs)]  # top right
    rect[2] = corners[np.argmax(sums)]  # bottom right
    rect[3] = corners[np.argmax(diffs)]  # bottom left
    # creates the original array of corner points
    src = np.array([rect[0], rect[1], rect[2], rect[3]], dtype='float32')

    # computes the side of the new transformation length we are looking for
    side = max([distance_between(rect[2], rect[1]),
                distance_between(rect[1], rect[3]),
                distance_between(rect[2], rect[3]),
                distance_between(rect[0], rect[1])])

    # creates the new array of corner points
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    # computes the new matrix for skewing the image to fit a square
    m = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(orig_image, m, (int(side), int(side)))


def grid(image):
    """
    Obtains the grid from the image.
    :param image: warped image to create grid from
    :return: a list of squares and their coordinates (top left, bottom right) which we can construct a square
    using the diagonal
    """
    grid_squares = []
    # image.shape returns a tuple of (num rows, num cols, channels)
    # since we know the warped image is a square, we can access the first element of the tuple
    side = image.shape[:1][0]
    # we are going to make our own grid out of the standard 9 x 9 Sudoku grid
    side = side / 9

    for j in range(9):
        for i in range(9):
            # computes the top left corner of a box
            p1 = (i * side, j * side)
            # computes the bottom right corner
            p2 = ((i + 1) * side, (j + 1) * side)
            grid_squares.append((p1, p2))

    return grid_squares


def extract_digits(image, squares):
    """
    Extracts the digit from each square
    :param image: the image
    :param squares: list of square coordinates (top left, bottom right) in the grid
    :return:
    """
    for i in squares:
        top_left = (int(i[0][0]), int(i[0][1]))
        bottom_right = (int(i[1][0]), int(i[1][1]))

        roi = image.copy()[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        roi_processed = process_image(roi, False)

        if DEBUG: show_image(roi_processed, 'roi processed')


def calc_rect_points(two_corners):
    """
    Helper function that computes coordinates of the four corners of a rectangle given two diagonal coordinates
    :param two_corners: tuple containing two coordinates of diagonal corners
    :return: list of coordinates of the four corners of a rectangle
    """
    x1, y1 = two_corners[0][0], two_corners[0][1]

    x2, y2 = two_corners[1][0], two_corners[1][1]

    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    xd, yd = (x1 - x2) / 2, (y1 - y2) / 2

    x3, y3 = xc - yd, yc + xd
    x4, y4 = xc + yd, yc - xd

    return [(x1, y1), (x3, y3), (x2, y2), (x4, y4)]


def main():
    image = cv2.imread('sudoku.jpeg')

    processed_image = process_image(image)

    # if DEBUG: show_image(processed_image, 'thresh')

    contours = find_contours(processed_image)

    corners = find_corners(contours)

    warped = perspective_transform(image, corners)

    # if DEBUG: show_image(warped, 'warped')

    squares = grid(warped)

    digits = extract_digits(warped, squares)

    model = cnn.load_model()


if __name__ == '__main__':
    main()
