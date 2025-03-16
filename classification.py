from ultralytics import YOLO
import numpy as np
import cv2
from utils.processing import *
import os

model = YOLO("yolov8n-face.pt")
card_size = (640, 320)

def convert_rec2corners(rec):
    return np.array([[rec[0], rec[1]], [rec[2], rec[1]], [rec[2], rec[3]], [rec[0], rec[3]]])

def remove_face(card: np.ndarray):
    result = model.predict(card, imgsz=640, conf=0.5, verbose=False)[0]
    boxes = [box.cpu().numpy() for box in result.boxes.xyxy]
    for rec in boxes:
        corners = convert_rec2corners(rec)
        corners = expand_corners(card.shape, corners, expand_rate=0.3)
        mask = np.ones_like(card) * 255
        mask = cv2.rectangle(mask, (int(corners[0,0]), int(corners[0,1])), (int(corners[2,0]), int(corners[2,1])), (0, 0, 0), -1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        card = cv2.bitwise_and(card, card, mask=mask)
    return card

def load_templates(template_dir):
    return [remove_face(load_image(os.path.join(template_dir, path), card_size)) for path in os.listdir(template_dir)]

def match(card, templates):
    return np.array([cv2.matchTemplate(card, template, cv2.TM_CCOEFF_NORMED)[0,0] for template in templates])

def classify(card, templates, threshold=0.7):
    scores = match(card, templates)
    if np.max(scores) > threshold:
        return np.argmax(scores)
    else:
        return None