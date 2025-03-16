from extraction import crop_image, document_detect
from classification import remove_face, classify
from ultralytics import YOLO
import cv2

MODEL_PATH = "localization/best_model/best.pt"
model = YOLO(MODEL_PATH)

def detect_boundingbox(image, model: YOLO):
    result = model(image)[0]
    corners = result.obb[0].xyxyxyxy[0].cpu().numpy()
    return corners

def detect_card(image, templates):
    corners = detect_boundingbox(image, model)
    cropped = crop_image(image, corners)
    cropped = cv2.resize(cropped, (640, 320))
    card = document_detect(cropped)
    card = cv2.resize(card, (640, 320))
    if card is None:
        return None, None
    removed_face = remove_face(card)
    label = classify(removed_face, templates)
    return card, label
