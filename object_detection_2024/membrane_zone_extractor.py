"""
This file contains functions to extract the test membrane zones, which will then be sent to the classifier.
"""

import math
import numpy as np
import torch
import cv2


def compute_rectangle(mask):
    """
    Function to compute the coordinates of the minimal (rotated) rectangle enclosing the mask.
    
    Args:
        mask (np.narray): image mask with shape (H, W) and with values [0, 1]
    
    Return:
        rect_coords (np.array): coordinates of rectangle [[xlb, ylb], [xrb, yrb], [xrt, yrt], [xlt, ylt]]
    """

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select only one contour, the one with maximum area
    contour = max(contours, key=cv2.contourArea)
    # Build minimal rectangular box containing the contour (it also returns angle...)
    rect = cv2.minAreaRect(contour)
    # Get coordinates of rectangle in a better format: list of 4 vertex points
    box_points = cv2.boxPoints(rect).astype('int')
    
    def sort_points(points):
        """
        Make sure the box points are always sorted in the same way
        """
    
        # Sort the points in box based on their y-coordinates
        points_ysorted = box_points[np.argsort(points[:, 1])]

        # Grab the bottommost and topmost points from the sorted y-coordinate points (NOTICE: (0, 0) is the topleft corner)
        topmost, bottommost  = points_ysorted[:2, :], points_ysorted[2:, :]

        # Sort the topmost coordinates according to their x-coordinates
        lefttop, righttop = topmost[np.argsort(topmost[:, 0]), :]

        # Sort the bottommost coordinates according to their x-coordinates
        leftbottom, rightbottom = bottommost[np.argsort(bottommost[:, 0]), :]
        
        return np.array([leftbottom, rightbottom, righttop,  lefttop], dtype='int')

    rect_coords = sort_points(box_points)
    
    return rect_coords

def compute_angle(rect_coords):
    """
    Compute angle of rotation of a given rectangle
    
    Args:
        rect_coords (np.ndarray): Assumes format [[xlb, ylb], [xrb, yrb], [xrt, yrt], [xlt, ylt]]
        
    Return:
        angle (int): angle of rotation.
    """
    
    # Decompose coordinates
    (xlb, ylb), (xrb, yrb), (xrt, yrt), (xlt, ylt) = rect_coords
    # Compute left and right angles
    left_angle = math.atan((xlb - xlt) / (ylb - ylt)) * (180 / math.pi)
    right_angle = math.atan((xrb - xrt) / (yrb - yrt)) * (180 / math.pi)
    # Calculate average angle
    angle = int(round((left_angle + right_angle) / 2))
                            
    return angle

def rotate_image(image, angle):
    """Function to rotate image in a specified (integer) angle"""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result