from ultralytics import YOLO
import numpy as np
import cv2
import itertools

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

def predict_corners(model: YOLO, image_dir):
    result = model.predict(image_dir, imgsz=320, conf=0.5, device='cpu')[0]
    corner = result.obb[0].xyxyxyxy[0].cpu().numpy()
    return corner

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

def extract_card(image, corners):
    if len(corners) < 4:
        return None
    corners = np.int32(corners)
    ordered_corners = order_points(corners)
    return four_point_transform(image, ordered_corners)

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

def draw_lines(image, lines):
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

def get_lines(edges):
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
            
    return new_lines

def document_detect(cropped_image):
    try:
        im = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        im = cv2.bitwise_not(im)
        saturation = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        saturation = cv2.medianBlur(saturation, 15)
        edges = auto_canny(saturation, 0.5)

        mask = np.ones_like(edges)
        height, width = edges.shape[:2]
        cv2.rectangle(mask, (width//8, height//8), (width*7//8, height*7//8), 0, -1)
        edges = apply_mask(edges, mask)
        lines = get_lines(edges)
        img = draw_lines(cropped_image.copy(), lines)
        intersection = find_intersections(lines, cropped_image)
        corners = intersection[:4]
        card = extract_card(cropped_image, corners)
    except:
        return None
    
    return card