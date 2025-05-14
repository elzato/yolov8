from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

# Untuk izin akses dari domain frontend (misal Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain spesifik di produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model YOLOv8 (pakai model bawaan atau custom kamu)
model = YOLO("yolov8n.pt")  # Ganti dengan "best.pt" jika pakai model sendiri

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = model(img)[0]
    detected_objects = []

    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        detected_objects.append(cls_name)

    return {"detections": detected_objects}
