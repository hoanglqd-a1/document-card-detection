import cv2
from card_detection import detect_card
from classification import load_templates
from ultralytics import YOLO

model = YOLO("localization/best_model/best.pt")

cap = cv2.VideoCapture(0)
template_dir = 'templates'
templates = load_templates(template_dir)


while True:
    Width, Height = 800, 600
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    card, label = detect_card(frame, templates)
    if card is not None:
        cv2.imshow('card', card)
        print(label)
    
    cv2.imshow('frame', frame)


cv2.destroyAllWindows()