import cv2
import numpy as np
import math


# Use a simple decision tree to find the shape of the contour
def decision_tree(contour):

    # Set up some variables

    # Contour measurements
    area = cv2.contourArea(contour)
    if area == 0:
        return None # Avoid ZeroDivisionError caused by noise
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(contour, True)
    # A polygon approximation of the contour - usually enough to resolve the shape, but not always
    poly = None  # The instantiation of poly is resolved in the decision tree

    # Tolerance constants
    min_ratio = 0.9  # The minimum aspect ratio a contour can have to be considered non-square
    min_convexity = 0.75  # The mininum area-to-convex-hull ratio that still counts as convex
    angle_tolerance = math.pi*15/180  # The tolerance in radians when checking for a right angle

    cv2.drawContours(img, [cv2.convexHull(contour)], 0, (0, 0, 255), 3)
    cv2.imshow('cool', img)
    cv2.waitKey(0)

    # Begin decision tree
    
    if area/cv2.contourArea(hull) < min_convexity:  # Is the shape concave?
        poly = cv2.approxPolyDP(contour, perimeter/40, True)
        num_points = poly.shape[0]
        if num_points == 10:
            return 'star'
        if num_points == 12:
            return 'cross'
        else:
            return 'undefined'
    else:
        # Since the shape is effectively convex, the polygon approximation should be fully convex
        poly = cv2.approxPolyDP(hull, perimeter/40, True)  
        # average_dist = 0
        # for i in range(hull.shape[0]):
            # average_dist += np.linalg.norm(hull[i][0]-hull[i-1][0])
            # cv2.circle(img, (hull[i][0][0], hull[i][0][1]), 3, (255, 0, 0))
            # cv2.imshow('cool', img)
            # cv2.waitKey(100)
        # print(average_dist/hull.shape[0])
        # print(cv2.arcLength(poly, True)/perimeter)
        if cv2.contourArea(poly)/area < 0.95:  # Is the shape non-polygonal?
            _, _, w, h = cv2.boundingRect(contour)
            if w/h > min_ratio and w/h < 1/min_ratio:  # Is the aspect ratio sufficiently close to 1?
                num_points = poly.shape[0]
                if num_points < 6:  # Because a quarter circle has straight edges, there should be fewer points
                    return 'quartercircle'
                else:
                    return 'circle'
            else:
                return 'semicircle'
        else:
            num_points = poly.shape[0]
            if num_points != 4:  # Is there only one shape with this number of points? (true for all but 4)
                if num_points == 3:
                    return 'triangle'
                if num_points == 5:
                    return 'pentagon'
                if num_points == 6:
                    return 'hexagon'
                if num_points == 7:
                    return 'heptagon'
                if num_points == 8:
                    return 'octagon'
                else:
                    return 'circle'
            else: 
                _, _, w, h = cv2.boundingRect(contour)
                if w/h > min_ratio and w/h < 1/min_ratio: # Is the aspect ratio sufficiently close to 1?
                    return 'square'
                else:
                    # Use vector math to find one angle of the polygon
                    [[[x1, y1]], [[x2, y2]], [[x3, y3]]] = poly[:3]
                    a_vec = np.array([x2-x1, y2-y1])
                    b_vec = np.array([x3-x2, y3-y2])
                    angle = math.acos(a_vec.dot(b_vec)/(np.linalg.norm(a_vec)*np.linalg.norm(b_vec)))
                    if abs(math.pi/2 - angle) < angle_tolerance:
                        return 'rectangle'
                    else:
                        return 'trapezoid'


# Hand drawn shape test
img = cv2.imread('./shape_test.png')
img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_bin, 240, 255, cv2.THRESH_BINARY_INV)
_, contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    print(decision_tree(contours[i]))
    cv2.drawContours(img, contours, i, (0, 255, 0), -1)
    cv2.imshow('cool', img)
    cv2.waitKey(0)
