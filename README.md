# 🏭 Warehouse Multi-Object Tracking System

Real-time detection and tracking system for warehouse surveillance using **YOLOv8 + OpenCV**.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Detected Classes](#detected-classes)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## ✅ Features

- 🔍 **3 Object Classes** detected — Person, Forklift, Parcel
- 🆔 **Unique Tracking IDs** — each object gets a consistent ID across frames
- 📊 **Live Analytics HUD** — real-time count displayed on screen
- ↕️ **Entry / Exit Counting** — counts when objects cross the virtual counting line
- 📝 **CSV Log Export** — per-frame data saved to a CSV file
- 🎥 **Video Save** — annotated output video can be saved

---

## 📁 Project Structure

```
warehouse_tracker/
├── main.py                  ← Entry point - run this file
├── requirements.txt         ← Python dependencies
├── utils/
│   ├── __init__.py          ← Empty file (required)
│   ├── track.py           ← Object tracking (IoU based)
│   ├── analysis.py         ← Entry/Exit counting
│   └── visualization.py        ← OpenCV drawing (boxes, labels, HUD)
└── output/                  ← CSV logs and videos are saved here
```

---

## 📦 Requirements

| Requirement | Version |
|---|---|
| Python | 3.9 or higher |
| ultralytics | >= 8.0.0 |
| opencv-python | >= 4.7.0 |
| numpy | >= 1.23.0 |
| pandas | >= 1.5.0 |
| matplotlib | >= 3.6.0 |

---

## ⚙️ Installation

**Step 1 — Create and activate a virtual environment**

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

**Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 — Verify the installation**

```bash
python -c "import ultralytics, cv2, numpy; print('All good!')"
```

---

## 🚀 Usage

**Run with webcam (default)**
```bash
python main.py --source 0
```

**Process a video file**
```bash
python main.py --source warehouse.mp4
```

**Use a higher accuracy model**
```bash
python main.py --source 0 --model yolov8s.pt
```

**Save the output video**
```bash
python main.py --source warehouse.mp4 --save
```

**RTSP IP Camera**
```bash
python main.py --source rtsp://192.168.1.100:554/stream
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index, video file path, or RTSP URL |
| `--model` | `yolov8n.pt` | YOLOv8 weights file |
| `--output` | `output` | Output folder for CSV and video |
| `--save` | False | Save the annotated output video |

> **Note:** On first run, YOLOv8 weights are automatically downloaded (~6MB for nano)

**To quit:** Press `Q` or `ESC`

---

## 📤 Output

### CSV Log
Saved to `output/log_TIMESTAMP.csv`:

| Column | Example | Description |
|---|---|---|
| frame | 142 | Frame number |
| time | 14:32:05 | Timestamp |
| track_id | 3 | Unique object ID |
| class | Person | Detected class |
| x1, y1, x2, y2 | 120,80,240,300 | Bounding box in pixels |
| confidence | 0.87 | Detection confidence (0–1) |
| event | ENTRY | Line cross event (ENTRY/EXIT/blank) |

### Terminal Summary
Displayed when the program exits:
```
=========== FINAL SUMMARY ===========
  Person     | Total:  12 | IN:  8 | OUT:  4
  Forklift   | Total:   3 | IN:  2 | OUT:  1
  Parcel     | Total:  27 | IN: 15 | OUT: 12
```

---

## 🎯 Detected Classes

The default YOLOv8 COCO model is used. The following proxy classes are mapped for warehouse use:

| Label | COCO Classes Used | Class IDs |
|---|---|---|
| Person | person | 0 |
| Forklift | truck | 7 |
| Parcel | handbag, suitcase, bottle, chair, laptop, cell phone, book | 24, 28, 39, 56, 63, 67, 73 |

> **For Better Accuracy:** Train a custom YOLOv8 model on warehouse-specific data.  
> Free datasets available at: [roboflow.com/search?q=warehouse](https://roboflow.com/search?q=warehouse)

---

## 🔧 Configuration

You can modify the following values at the top of `main.py`:

```python
CONF_THRESHOLD = 0.30   # Detection confidence (lower = more detections, more false positives)
IOU_THRESHOLD  = 0.45   # NMS threshold
LINE_POSITION  = 0.5    # Counting line position (0.0=top, 1.0=bottom)
```

### Model Size vs Speed (CPU)

| Model | File | Speed | Accuracy |
|---|---|---|---|
| YOLOv8 Nano | `yolov8n.pt` | Fastest (~20 FPS) | Lower |
| YOLOv8 Small | `yolov8s.pt` | Fast (~12 FPS) | Good ✅ |
| YOLOv8 Medium | `yolov8m.pt` | Moderate (~6 FPS) | Better |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: utils.tracker` | File must be named `tracker.py` (not `tracer.py`) |
| `ModuleNotFoundError: utils.visualizer` | File must be named `visualizer.py` (not `viz.py`) |
| Webcam not opening | Try `--source 1` or `--source 2` |
| Only Person being detected | Check all class IDs in `WAREHOUSE_CLASSES` |
| Running very slowly | Use `yolov8n.pt` or resize the input frames |
| `venv\Scripts\activate` error | Run PowerShell as Administrator |

---

## 📄 License

This project is intended for educational and surveillance purposes.

---

*Built with YOLOv8 + OpenCV + Python*
