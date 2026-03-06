from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='ultralytics/datasets/pdd/data.yaml', epochs=50, batch=16)