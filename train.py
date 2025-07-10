from ultralytics import YOLO

def train_balanced():
    model = YOLO("yolov8s.pt")  # small model

    model.train(
        data="data.yaml",     # your dataset config path
        epochs=25,            # moderate epochs for accuracy + speed
        imgsz=640,            # image size
        batch=8,              # smaller batch for CPU
        device="cpu"          # run on CPU since no GPU available
    )

if __name__ == "__main__":
    train_balanced()
