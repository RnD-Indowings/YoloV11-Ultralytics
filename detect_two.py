# Indo Wings CyberOne Mini Autonomous Strike Kamikaze

import argparse, csv, os, sys, time, platform
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils.dataloaders import LoadImages, LoadStreams  # type: ignore

from dronekit import connect, VehicleMode
from pymavlink import mavutil

# ===================== LOGS =====================
os.makedirs("logs", exist_ok=True)
os.makedirs("strikes", exist_ok=True)
log_filename = datetime.now().strftime("logs/output_%Y%m%d_%H%M%S.txt")
sys.stdout = open(log_filename, "a")
sys.stderr = sys.stdout

# ================== DRONE CONNECTION ==================
vehicle = connect('127.0.0.1:14550', wait_ready=True)
vehicle.mode = VehicleMode("GUIDED")
while vehicle.mode.name != "GUIDED": time.sleep(1)
vehicle.armed = True
while not vehicle.armed: time.sleep(1)
vehicle.simple_takeoff(15.0)
time.sleep(15)  # stabilize at 15 m

def send_velocity_body(vehicle, vx, vy, vz):
    """Send velocity command in drone body frame (forward, right, down)."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,  # enable only velocity
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# ================== MAIN RUN ==================
def run(
    vehicle,
    weights="yolov11n.pt",
    source=0,
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    save_txt=False,
    save_crop=False,
    save_csv=False,
    project="runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    vx_const=5.0,
    Kp_y=0.02,
    Kp_z=0.03,
    max_vel_y=3.0,
    max_vel_z=4.0,
    view_img=False
):
    source = str(source)
    save_img = not save_txt

    # Load YOLO model
    model = YOLO(weights)

    # Dataset
    if source.isdigit() or source.lower().startswith(("rtsp://", "http://", "https://")):
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Save directories
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=exist_ok)
    (save_dir / "labels").mkdir(exist_ok=True)
    if save_crop:
        (save_dir / "crops").mkdir(exist_ok=True)
    csv_path = save_dir / "predictions.csv"

    def write_to_csv(image_name, prediction, confidence):
        data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    # Video recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join('strikes', f"recording_{timestamp}.mp4")
    video_writer = None
    fps = 30

    # ========== PROCESS FRAMES ==========
    for path, im, im0s, vid_cap, s in dataset:
        results = model.predict(im, imgsz=imgsz, conf=conf_thres, iou=iou_thres, max_det=max_det)
        for r in results:
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=line_thickness)
            H, W = im0.shape[:2]
            cx, cy = W//2, H//2

            if r.boxes is not None:
                best, min_d = None, float('inf')
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    # Only human
                    if cls != 7: continue
                    px, py = (x1+x2)/2, (y1+y2)/2
                    d = np.hypot(px-cx, py-cy)
                    if d < min_d: min_d, best = d, (px, py, x1, y1, x2, y2, conf, cls)

                if best is not None:
                    bbox_cx, bbox_cy, x1, y1, x2, y2, conf, cls = best
                    err_x, err_y = bbox_cx - cx, bbox_cy - cy
                    vx = vx_const
                    vy = float(np.clip(Kp_y*err_x, -max_vel_y, max_vel_y))
                    vz = float(np.clip(Kp_z*err_y, -max_vel_z, max_vel_z))
                    send_velocity_body(vehicle, vx, vy, vz)

                    label = f"{r.names[cls]} {conf:.2f}" if not hide_conf else r.names[cls]
                    annotator.box_label([x1, y1, x2, y2], label, color=colors(cls, True))

                    if save_csv:
                        write_to_csv(Path(path).name, r.names[cls], f"{conf:.2f}")
                    if save_txt:
                        txt_file = save_dir / "labels" / (Path(path).stem + ".txt")
                        coords = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
                        with open(txt_file, "a") as f:
                            f.write(f"{cls} " + " ".join([f"{c:.6f}" for c in coords]) + "\n")
                    if save_crop:
                        save_one_box([x1, y1, x2, y2], im0, file=save_dir / "crops" / r.names[cls] / f"{Path(path).stem}.jpg", BGR=True)

            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(Path(path)), im0)
                cv2.waitKey(1)
            if save_img:
                cv2.imwrite(save_dir / Path(path).name, im0)
                if vid_cap is not None:
                    if video_writer is None:
                        h, w = im0.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                    video_writer.write(im0)

    if video_writer is not None:
        video_writer.release()
        print(f"[INFO] Recording saved to {video_path}")
    cv2.destroyAllWindows()


# ================== ARGPARSE ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model", type=str, default="yolov11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--noshow", action="store_true")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    run(
        vehicle=vehicle,
        source=src,
        weights=args.model,
        conf_thres=args.conf,
        save_txt=not args.nosave,
        view_img=not args.noshow
    )
