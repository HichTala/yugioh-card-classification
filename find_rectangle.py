import cv2
import numpy as np


def find_rectangles(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Convert the image to grayscale
    gray = image[:, :, 2]
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply edge detection using Canny
    edges = cv2.Canny(image, 50, 150)
    # Display the images
    cv2.imshow('Lines', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print("No lines detected.")
        return

    # Draw the lines on a copy of the original image
    lines_image = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Find rectangles among the detected lines
    rectangles = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            for k in range(j + 1, len(lines)):
                rect = is_rectangle(lines[i][0], lines[j][0], lines[k][0])
                if rect is not None:
                    rectangles.append(rect)

    # Draw rectangles on a copy of the original image
    rectangles_image = image.copy()
    for rect in rectangles:
        cv2.drawContours(rectangles_image, [np.array(rect)], 0, (0, 255, 0), 2)

    # Display the images
    cv2.imshow('Lines', lines_image)
    cv2.imshow('Rectangles', rectangles_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def is_rectangle(line1, line2, line3):
    # Check if the lines form a rectangle
    epsilon = 10
    angle_thresh = 10

    def get_angle(line):
        x1, y1, x2, y2 = line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle

    angles = [get_angle(line1), get_angle(line2), get_angle(line3)]

    if (
            abs(angles[0] - angles[1]) < angle_thresh
            and abs(angles[1] - angles[2]) < angle_thresh
            and abs(angles[2] - angles[0]) < angle_thresh
    ):
        # Sort the lines based on length
        lines = [line1, line2, line3]
        lines.sort(key=lambda l: np.linalg.norm(np.array([l[0], l[1]]) - np.array([l[2], l[3]])))

        # Check if opposite sides have similar lengths
        side1 = np.linalg.norm(np.array([lines[0][0], lines[0][1]]) - np.array([lines[1][2], lines[1][3]]))
        side2 = np.linalg.norm(np.array([lines[1][0], lines[1][1]]) - np.array([lines[0][2], lines[0][3]]))
        side3 = np.linalg.norm(np.array([lines[1][2], lines[1][3]]) - np.array([lines[2][0], lines[2][1]]))
        side4 = np.linalg.norm(np.array([lines[2][2], lines[2][3]]) - np.array([lines[0][0], lines[0][1]]))

        if (
                abs(side1 - side3) < epsilon
                and abs(side2 - side4) < epsilon
                and abs(side1 - side2) < epsilon
        ):
            return np.array([
                [lines[0][0], lines[0][1]],
                [lines[1][2], lines[1][3]],
                [lines[1][0], lines[1][1]],
                [lines[2][2], lines[2][3]]
            ], dtype=np.int32)

    return None


# Example usage
find_rectangles('/home/hicham/Documents/draw/ROI/frame_9999_box_9_Dark-Magician-Girl-0-38033121.png')
