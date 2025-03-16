from ultralytics import YOLO

# search_space = {
#     "lr0": (1e-5,1e-2),
#     "degrees": (0, 45),
#     "translate": (0, 0.2),
#     "scale": (0.0, 0.2),
#     "shear": (0, 10),
#     "perspective": (0, 0.2),
# }

if __name__ == '__main__':
    model = YOLO("yolo_model/yolov8s-obb.pt")
    model.train(data="data.yaml", epochs=100, batch=16, imgsz=640, patience=10, degrees=30, flipud=0.5, fliplr=0.5, lr0=1e-2, device=0)
    # model.tune(data="data.yaml", epochs=30, space=search_space, plots=False, save=False, val=False)