import argparse
import yaml
from ultralytics import YOLO
import cv2
import os

def load_classes_from_yaml(data_yaml):
    """Load class names from data.yaml."""
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"{data_yaml} not found.")
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    return list(data.get("names", {}).keys()) if isinstance(data.get("names"), dict) else data.get("names", [])

def run(source=4, model_path="yolo11l.pt", conf=0.80, save=True, show=True, data_yaml=None):
    # Load YOLOv11 model
    model = YOLO(model_path)

    # Load dataset classes if provided
    classes = None
    if data_yaml:
        classes = load_classes_from_yaml(data_yaml)
        print(f"Using classes from {data_yaml}: {classes}")

    # Run prediction (stream=True for live webcam/video frame by frame)
    results = model.predict(source=source, conf=conf, stream=True, save=save, classes=[0])

    # Iterate through results
    for r in results:
        frame = r.plot()  # get frame with boxes drawn
        if show:
            cv2.imshow("YOLOv11 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="4", help="Source (0=webcam, path to image/video)")
    parser.add_argument("--model", type=str, default="yolo11l.pt", help="Path to YOLOv11 model")
    parser.add_argument("--conf", type=float, default=0.80, help="Confidence threshold")
    parser.add_argument("--nosave", action="store_true", help="Do not save results")
    parser.add_argument("--noshow", action="store_true", help="Do not display popup window")
    parser.add_argument("--data", type=str, default="/home/udit/ultralytics/ultralytics/cfg/datasets/coco128.yaml", help="Path to data.yaml for class names")
    args = parser.parse_args()

    # Convert webcam source from string to int
    source = int(args.source) if args.source.isdigit() else args.source

    run(
        source=source,
        model_path=args.model,
        conf=args.conf,
        save=not args.nosave,
        show=not args.noshow,
        data_yaml=args.data
    )
