# 🚁 YOLOv11 Aerial Vehicle Detection System

End-to-end Computer Vision project designed to detect and track aerial vehicles (**airplanes, UAVs/drones, and helicopters**) using **Ultralytics YOLOv11** and **Roboflow**.

---

## 🚀 Features

* **Automated Dataset Download:** Integrates directly with the Roboflow API to fetch and structure datasets in YOLOv11 format.
* **Custom Model Training (`main.py`):** Trains a customized YOLOv11 medium (`yolo11m.pt`) model optimized for aerial target detection.
* **Multi-Modal Testing Interface:** An interactive CLI menu that supports:
  1. **Batch Image Testing:** Processes multiple test images and saves prediction results.
  2. **Real-time Webcam Inference:** Low-latency live detection via connected camera feed.
  3. **Video Processing:** Batch processes `.mp4` video files with bounding box visualization and saving.

---

## 🛠️ Tech Stack & Requirements

* **Python** (3.8+)
* **Ultralytics YOLOv11**
* **Roboflow SDK**
* **OpenCV**
* **PyTorch** (CUDA-compatible GPU recommended for training)

---

## 📂 Project Structure

```text
yolov11-aerial-vehicle-detection/
│
├── model.py             # Dataset downloading and model training script
├── prediction.py   # Interactive test script (Image, Video, Webcam)
├── image/              # Directory for test images (.jpg)
├── video/              # Directory for test videos (.mp4)
└── runs/               # Training outputs, weights, and prediction results
