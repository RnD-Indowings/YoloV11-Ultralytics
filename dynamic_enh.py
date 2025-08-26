import argparse
from ultralytics import YOLO
import cv2
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from dronekit import connect, VehicleMode
from pymavlink import mavutil
import math
import time
import sys , os
from datetime import datetime

# -------------------- DRONE CONNECTION --------------------
vehicle = connect('127.0.0.1:14550', wait_ready=True)
vehicle.mode = VehicleMode("GUIDED")
while vehicle.mode.name != "GUIDED":
    time.sleep(1)
vehicle.armed = True
while not vehicle.armed:
    time.sleep(1)
vehicle.simple_takeoff(15.0)
time.sleep(15.0)  # allow time to stabilize at 4 m

def send_velocity_body(vehicle, vx, vy, vz):
    """Send velocity command in the drone's body frame (forward, right, down)."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,  # only velocity enabled
        0, 0, 0,             # no position
        vx, vy, vz,          # velocity components
        0, 0, 0,             # no acceleration
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


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
            
    # Control parameters
    vx_const  = 5.0    # constant forward speed (m/s)
    Kp_y      = 0.02   # lateral gain (m/s per pixel vert error)
    Kp_z      = 0.03   # vertical gain (m/s per pixel horiz error)
    max_vel_y = 3.0    # max lateral speed
    max_vel_z = 4.0    # max vertical speed
    
    # ===========Image center==============#
    H = 640
    W = 480
    cx, cy = W//2, H//2
    if len(det):
        ##-- HUMAN DETECTION ONLY --##
        det = det[det[:, 5] == 0]
        # Rescale boxes from img_size to im0 size
        
        xyxy = det[0, :4]       # x1, y1, x2, y2
        #conf = det[0, 4]        # confidence
        bbox_cx = (xyxy[0] + xyxy[2]) / 2
        bbox_cy = (xyxy[1] + xyxy[3]) / 2

        err_x = bbox_cx - cx
        err_y = bbox_cy - cy

        # --- 6. Map to fixed vx and dynamic vy, vz ---
        vx = vx_const
        vy = float(np.clip((Kp_y * err_x).cpu().item(), -max_vel_y, max_vel_y))

        vz = float(np.clip((Kp_z * err_y).cpu().item(), -max_vel_z, max_vel_z))

        # --- 7. Send velocity command to strike ---
        send_velocity_body(vehicle, vx, vy, vz)
        
        altitude=vehicle.location.global_relative_frame.alt
        

        vx, vy, vz = vehicle.velocity  # (North, East, Down)
        print(f"Live Velocity: vx = {vx:.2f} m/s, vy = {vy:.2f} m/s")
        
        print(f"Velocity && Alt:  vz={vz}, {altitude}")
        
        
        if not vehicle.armed:
            print("[WARNING] Drone disarmed! Stopping execution.")
            vehicle.mode = VehicleMode("LOITER")  # Optional safe fallback
            sys.exit(0)  # Hard stop execution 
                


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
