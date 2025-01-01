from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import docdetect
from card_segmentation.utils import processing
import os
import shutil
import itertools

def predict_corners(model, image_dir):
    result = model.predict(image_dir, imgsz=320, conf=0.5, device='cpu')[0]
    corner = result.obb[0].xyxyxyxy[0].cpu().numpy()
    return corner
    
def crop_image(raw_image, corners, expand_rate=0.05):
    corners = processing.expand_corners(raw_image.shape, corners, expand_rate)
    return processing.four_point_transform(raw_image, corners)

def extract_card(image, corners):
    if len(corners) < 4:
        return None
    corners = np.int32(corners)
    ordered_corners = processing.order_points(corners)
    return processing.four_point_transform(image, ordered_corners)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper, L2gradient = True)

    # return the edged image
    return edged

def find_intersections(lines, im):
    height, width = im.shape[:2]
    intersections = []
    for line1, line2 in itertools.permutations(lines, 2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        a = [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]
        b = [[rho1], [rho2]]
        try:
            # aX = b, solve for x
            x, y = np.round(np.linalg.solve(a, b))
            x, y = int(x[0]), int(y[0])
            if -width/10 < x < width + width/10 and -height/10 < y < height + height/10:
                if (x, y) not in intersections: intersections.append((x, y))

        except:
            pass
    return intersections

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def document_detect(cropped_image):
    try:
        im = cv2.bitwise_not(cropped_image)
        saturation = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        saturation = cv2.medianBlur(saturation, 15)
        edges = auto_canny(saturation, 0.5)

        mask = np.ones_like(edges)
        height, width = edges.shape[:2]
        cv2.rectangle(mask, (width//8, height//8), (width*7//8, height*7//8), 0, -1)
        edges = apply_mask(edges, mask)

        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        lines = lines[:, 0, :]
        lines = [(line[0], line[1]) for line in lines]
        line_groups = []
        for line in lines:
            if np.pi/6 < line[1] < np.pi/3 or np.pi*2/3 < line[1] < np.pi*5/6:
                continue
            if line[1] > np.pi*5/6:
                line = (-line[0], line[1] - np.pi)
            for group in line_groups:
                mean_rho = np.mean([line[0] for line in group])
                mean_theta = np.mean([line[1] for line in group])
                if abs(line[0] - mean_rho) < 50 and abs(line[1] - mean_theta) < 0.3:
                    group.append(line)
                    break
            else:
                line_groups.append([line])
                
        new_lines = []
        for group in line_groups:
            rho = np.mean([line[0] for line in group])
            theta = np.mean([line[1] for line in group])
            new_lines.append((rho, theta))
                
        lines = new_lines
        intersection = find_intersections(lines, cropped_image)
        corners = intersection[:4]
        card = extract_card(cropped_image, corners)
    # quadrilaterals = docdetect.find_quadrilaterals(intersection)
    # frame = docdetect.draw(quadrilaterals, cropped_image)
    # return frame
    except:
        return None
    return card