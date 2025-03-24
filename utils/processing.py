# https://github.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/blob/master/four_point_object_extractor.py

import cv2
import numpy as np
from PIL import Image

def load_image(image_path, image_size: tuple[int, int]):
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image)
    return image

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# def check_validity(points, image_size):
#     for i in range(4):
#         for j in range(i+1, 4):
#             if abs(points[i][0][0] - points[j][0][0]) < image_size[0]//8 and abs(points[i][0][1] - points[j][0][1]) < image_size[1]//8:
#                 return False
            
#     return True

def find_contours(image, thickness=3):
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, 255, thickness)
    return contour_image, contours, hierarchy

def get_corners(raw_image, mask_image):
    _, contours, _ = find_contours(mask_image)

    image_size = mask_image.shape

    assert raw_image is not None
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    screenCntList = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        screenCnt = approx

        if (len(screenCnt) == 4):
            screenCntList.append(screenCnt)

    validScreenCntList = [points for points in screenCntList if check_validity(points, image_size)]
    return validScreenCntList

def expand_corners(shape, corners, expand_rate=0.05):
    center = np.mean(corners, axis=0)
    new_corners = center + (corners - center) * (1 + expand_rate)
    for i in range(4):
        new_corners[i, 0] = np.clip(new_corners[i, 0], 0, shape[1] - 1)
        new_corners[i, 1] = np.clip(new_corners[i, 1], 0, shape[0] - 1)
    return new_corners

def crop_image(raw_image, corners, expand_rate=0.05):
    corners = expand_corners(raw_image.shape, corners, expand_rate)
    return four_point_transform(raw_image, corners)
