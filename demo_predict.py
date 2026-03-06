from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
yolo.predict(source="https://ultralytics.com/images/bus.jpg", show=True, save=True)

