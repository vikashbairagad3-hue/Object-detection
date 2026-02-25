import cv2
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from utils.track import SimpleTracker      
from utils.analysis import Analytics
from utils.visualization import draw_frame     

WAREHOUSE_CLASSES = {
    0:  "Person",
    7:  "Forklift",   
    24: "Parcel",     
    28: "Parcel",      
    39: "Parcel",      
    56: "Parcel",      
    63: "Parcel",      
    67: "Parcel",     
    73: "Parcel",      
}

CLASS_COLORS = {
    "Person":   (0, 255, 0),
    "Forklift": (0, 165, 255),
    "Parcel":   (255, 0, 0),
}

CONF_THRESHOLD = 0.30   
IOU_THRESHOLD  = 0.45
LINE_POSITION  = 0.5  


def run(source, model_path, output, show, save):

    Path(output).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * LINE_POSITION)

    print(f"[INFO] Resolution: {width}x{height} | FPS: {fps:.1f}")

    writer = None
    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{output}/tracked_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving video to: {out_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{output}/log_{ts}.csv"
    csvf = open(csv_path, "w", newline="")
    log = csv.writer(csvf)
    log.writerow([
        "frame", "time", "track_id", "class",
        "x1", "y1", "x2", "y2", "confidence", "event"
    ])

    print(f"[INFO] Logging CSV to: {csv_path}")

    tracker = SimpleTracker(max_age=30, min_hits=2)
    analytics = Analytics()

    frame_id = 0
    start_time = time.time()

    print("[INFO] Press 'Q' or 'ESC' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended")
                break

            frame_id += 1
            now_time = datetime.now().strftime("%H:%M:%S")

            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=list(WAREHOUSE_CLASSES.keys()),  # fix 4: saare IDs pass kiye
                verbose=False,
            )[0]

            detections = []
            if results.boxes is not None:
                for b in results.boxes:
                    cls_id = int(b.cls[0])
                    if cls_id in WAREHOUSE_CLASSES:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        conf = float(b.conf[0])
                        detections.append([x1, y1, x2, y2, conf, cls_id])

            tracks = tracker.update(detections)

            for t in tracks:
                x1, y1, x2, y2, tid, cls_id, conf = t
                label = WAREHOUSE_CLASSES.get(cls_id, "Unknown")

                event = analytics.update(
                    tid,
                    label,
                    (x1 + x2) // 2,
                    (y1 + y2) // 2,
                    line_y
                )

                log.writerow([
                    frame_id, now_time, tid, label,
                    x1, y1, x2, y2, f"{conf:.2f}", event or ""
                ])

            live_fps = frame_id / max(time.time() - start_time, 1e-6)

            annotated = draw_frame(
                frame,
                tracks,
                WAREHOUSE_CLASSES,
                CLASS_COLORS,
                analytics.get_stats(),
                line_y,
                live_fps,
                frame_id
            )

            if writer:
                writer.write(annotated)

            cv2.imshow("Warehouse Multi-Object Tracking", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] User exited")
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        csvf.close()
        cv2.destroyAllWindows()

        print("\n Final Result")
        stats = analytics.get_stats()
        for cls, d in stats["class_stats"].items():
            print(f"  {cls:10s} | Total:{d['total']:4d} | IN:{d['entries']:3d} | OUT:{d['exits']:3d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 = webcam | video file | RTSP")
    parser.add_argument("--model",  default="yolov8n.pt", help="YOLOv8 model")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--save",   action="store_true", help="Save output video")
    parser.add_argument("--show",   action="store_true", help="(kept for compatibility)")

    args = parser.parse_args()

    run(
        source=args.source,
        model_path=args.model,
        output=args.output,
        show=True,
        save=args.save
    )
