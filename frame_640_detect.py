import argparse
from ultralytics import YOLO
import cv2


def run(source=0, model_path="yolo11l.pt", conf=0.25, save=True, show=True):
    # Load YOLOv11 model
    model = YOLO(model_path)

    # Open video source manually if it's a webcam
    cap = None
    if isinstance(source, int):  # webcam
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        # Run prediction on frames from the capture
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf, save=save, classes=[0])
            for r in results:
                out_frame = r.plot()
                if show:
                    cv2.imshow("YOLOv11 Detection", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
    else:
        # For video/image file input
        results = model.predict(source=source, conf=conf, save=save, classes=[0])
        for r in results:
            out_frame = r.plot()
            if show:
                cv2.imshow("YOLOv11 Detection", out_frame)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Source (0=webcam, path to image/video)")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Path to YOLOv11 model")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold")
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
