import argparse
import os
import sys
import time
import math
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from dronekit import connect, VehicleMode
from pymavlink import mavutil


# ------------------------ LOG SAVER ----------------------------- #
os.makedirs("logs", exist_ok=True)
os.makedirs("strikes", exist_ok=True)

# Timestamped log filename
log_filename = datetime.now().strftime("logs/output_%Y%m%d_%H%M%S.txt")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Redirect stdout and stderr to the file
log_fp = open(log_filename, "a", buffering=1)
sys.stdout = log_fp
sys.stderr = log_fp

time.sleep(0)


# -------------------- DRONE CONNECTION -------------------- #
def connect_and_takeoff(connection_str: str, target_alt: float = 0.0):
    print(f"[INFO] Connecting to vehicle at {connection_str} ...")
    vehicle = connect(connection_str, wait_ready=True)

    print("[INFO] Setting GUIDED mode ...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.2)

    print("[INFO] Arming motors ...")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(0.2)

    print(f"[INFO] Taking off to {target_alt:.1f} m ...")
    vehicle.simple_takeoff(target_alt)

    # Wait until the vehicle reaches ~95% of target altitude
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"[TAKEOFF] Alt: {alt:.2f} m")
        if alt is not None and alt >= 0.00 * target_alt:
            print("[INFO] Target altitude reached (≈95%).")
            break
        time.sleep(0.5)

    return vehicle


def send_velocity_body(vehicle, vx, vy, vz):
    """Send velocity command in the drone's body frame (forward, right, down)."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,  # only velocity enabled
        0, 0, 0,             # x, y, z positions (not used)
        float(vx), float(vy), float(vz),  # vx, vy, vz in m/s
        0, 0, 0,             # ax, ay, az (not used)
        0, 0                  # yaw, yaw_rate (not used)
    )
    vehicle.send_mavlink(msg)
    try:
        vehicle.flush()
    except Exception:
        # Some DroneKit versions don’t require/implement flush
        pass


def run(source=0, model_path="yolo11l.pt", conf=0.25, save=True, show=True):
    # -------------------- Connect & Takeoff -------------------- #
    vehicle = connect_and_takeoff("127.0.0.1:14550", target_alt=15.0)

    # -------------------- Load YOLO model -------------------- #
    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # Use stream=True to iterate frame-by-frame; classes=[0] to detect persons only.
    results_gen = model.predict(source=source, conf=conf, stream=True, save=save, classes=[0])

    # -------------------- Control Parameters -------------------- #
    vx_const  = 5.0    # constant forward speed (m/s)
    Kp_y      = 0.02   # lateral gain (m/s per pixel horizontal error)
    Kp_z      = 0.03   # vertical gain (m/s per pixel vertical error)
    max_vel_y = 3.0    # max lateral speed
    max_vel_z = 4.0    # max vertical speed
    dt        = 0.10   # control loop period ~10 Hz

    last_cmd_time = time.time()

    print("[INFO] Entering detection + control loop ... (press 'q' to quit window)")
    try:
        for r in results_gen:
            # If disarmed mid-flight, fail-safe to LOITER and stop
            if not vehicle.armed:
                print("[WARNING] Drone disarmed! Switching to LOITER and exiting loop.")
                vehicle.mode = VehicleMode("LOITER")
                break

            # Visualized frame
            # r.plot() returns the frame with boxes drawn; r.orig_img is the raw frame
            frame = r.plot()
            H, W = frame.shape[:2]
            cx, cy = W // 2, H // 2

            # Extract detections robustly
            # r.boxes.* are torch tensors
            person_available = (r.boxes is not None) and (r.boxes.cls is not None) and (len(r.boxes) > 0)

            if person_available:
                # Filter class==0 (person), though we also passed classes=[0] to predict
                cls_np  = r.boxes.cls.int().cpu().numpy()
                conf_np = r.boxes.conf.cpu().numpy()
                xyxy_np = r.boxes.xyxy.cpu().numpy()

                mask = (cls_np == 0)
                if mask.any():
                    confs = conf_np[mask]
                    boxes = xyxy_np[mask]

                    # Pick highest confidence person
                    idx = int(np.argmax(confs))
                    x1, y1, x2, y2 = boxes[idx]
                    bbox_cx = (x1 + x2) / 2.0
                    bbox_cy = (y1 + y2) / 2.0
                    best_conf = float(confs[idx])

                    # Pixel error (image frame: +x right, +y down)
                    err_x = bbox_cx - cx   # right positive
                    err_y = bbox_cy - cy   # down positive

                    # Control (Body-NED: +X forward, +Y right, +Z down)
                    vx = vx_const
                    vy = float(np.clip(Kp_y * err_x, -max_vel_y, max_vel_y))
                    vz = float(np.clip(Kp_z * err_y, -max_vel_z, max_vel_z))

                    # Throttle command rate
                    now = time.time()
                    if now - last_cmd_time >= dt:
                        send_velocity_body(vehicle, vx, vy, vz)
                        last_cmd_time = now

                        alt = vehicle.location.global_relative_frame.alt
                        vx_meas, vy_meas, vz_meas = vehicle.velocity
                        print(
                            f"[CONTROL] det_conf={best_conf:.2f} | "
                            f"err_x={err_x:.1f}px err_y={err_y:.1f}px | "
                            f"cmd vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} | "
                            f"meas vx={vx_meas:.2f} vy={vy_meas:.2f} vz={vz_meas:.2f} | Alt={alt:.2f}"
                        )
                else:
                    # No person class after mask -> hover forward only or consider hold
                    pass

            # Show window if requested
            if show:
                cv2.imshow("CB-One-T-M V-11 Kamikaze Drone Dt", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] 'q' pressed. Exiting loop.")
                    break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Exiting loop.")
    finally:
        # Fail-safe: switch to LOITER before exit
        try:
            print("[INFO] Switching to LOITER for safe hover.")
            vehicle.mode = VehicleMode("LOITER")
        except Exception as e:
            print(f"[WARN] Could not set LOITER: {e}")

        # Close any OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Don’t close vehicle immediately; give a moment to apply mode
        time.sleep(1.0)
        try:
            vehicle.close()
        except Exception:
            pass

        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                        help="Source (0=webcam, path to image/video/rtsp)")
    parser.add_argument("--model", type=str, default="yolo11l.pt",
                        help="Path to YOLOv11 model (e.g., yolo11n.pt / yolo11l.pt)")
    parser.add_argument("--conf", type=float, default=0.60,
                        help="Confidence threshold")
    parser.add_argument("--nosave", action="store_true",
                        help="Do not save results")
    parser.add_argument("--noshow", action="store_true",
                        help="Do not display popup window")
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
