# detect.py for YOLOv11 (re-implemented from YOLOv5 style)

# python3 yolov5_modified_detect.py       for running down with pop-up screen in yolov5 style


import argparse
from ultralytics import YOLO
import cv2

def run(
    weights="yolo11n.pt",
    source="4",
    data="/home/udit/ultralytics/ultralytics/cfg/datasets/coco128.yaml",        
    imgsz=640,
    conf_thres=0.80,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    project="runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    vid_stride=1
):
    # Load model
    model = YOLO(weights)

    # Run prediction
    results = model.predict(
        source=source,
        data=data,
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        save=not nosave,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        classes=classes,
        agnostic_nms=agnostic_nms,
        project=project,
        name=name,
        exist_ok=exist_ok,
        vid_stride=vid_stride,
        stream=True
    )

    # Iterate through results (for viewing)
    for r in results:
        im_array = r.plot(
            line_width=line_thickness,
            labels=not hide_labels,
            conf=not hide_conf
        )  # returns BGR numpy array

        if view_img:
            cv2.imshow("YOLOv11 Detection", im_array)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11x.pt", help="model path")
    parser.add_argument("--source", type=str, default="4", help="file/dir/URL/glob, 0 for webcam")
    parser.add_argument("--data", type=str, default="/home/udit/ultralytics/ultralytics/cfg/datasets/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size")
    parser.add_argument("--conf-thres", type=float, default=0.80, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--view-img", action="store_true", default=True, help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, default=0,  help="filter by class: --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness")
    parser.add_argument("--hide-labels", action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", action="store_true", help="hide confidences")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    return parser.parse_args()


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
