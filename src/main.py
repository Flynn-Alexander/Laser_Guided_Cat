from ultralytics import YOLO

# Load a model
model = YOLO("models/YOLO/yolov8n-pose.pt")  # load a pretrained model (recommended for training)
results = model.predict(source="1", show=True)  # predict on the webcam stream
