import argparse
from ultralytics import YOLO
import cv2

def run(source=0, model_path="yolo11l.pt", conf=0.25, save=True, show=True):
    # Load YOLOv11 model
    model = YOLO(model_path)

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
    parser.add_argument("--source", type=str, default="0", help="Source (0=webcam, path to image/video)")
    parser.add_argument("--model", type=str, default="yolo11l.pt", help="Path to YOLOv11 model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nosave", action="store_true", help="Do not save results")
    parser.add_argument("--noshow", action="store_true", help="Do not display popup window")
    args = parser.parse_args()

    # Convert webcam source from string to int
    source = int(args.source) if args.source.isdigit() else args.source

    run(
        source=source,
        model_path=args.model,
        conf=args.conf,
        save=not args.nosave,
        show=not args.noshow
    )
