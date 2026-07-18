# -*- coding: utf-8 -*-

import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO, RTDETR
import uvicorn

app = FastAPI(
    title="Bone Fracture Detection API",
    description="Бэкенд на FastAPI для детекции переломов с поддержкой YOLOv11 и RT-DETR",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Предзагрузка моделей
models = {
    "YOLOv11": YOLO("yolo11n.pt"),
    "RT-DETR": RTDETR("rtdetr-l.pt")
}

class DetectionResult(BaseModel):
    box: List[int]
    confidence: float
    label: str

class PredictResponse(BaseModel):
    detections: List[DetectionResult]

def preprocess_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    return img, enhanced_bgr

@app.get("/")
def read_root():
    return {
        "status": "Бэкенд успешно запущен и работает"
        # "interactive_docs": "Перейдите на http://127.0.0 для тестирования API"
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(
        image: UploadFile = File(...),
        model_type: str = Form(...)
):
    if model_type not in models:
        raise HTTPException(status_code=400, detail=f"Модель '{model_type}' не поддерживается.")

    try:
        img_bytes = await image.read()
        _, processed_img = preprocess_image(img_bytes)
    finally:
        await image.close()

    if processed_img is None:
        raise HTTPException(status_code=400, detail="Файл не является валидным изображением.")

    model = models[model_type]
    results = model(processed_img, imgsz=640)

    detections = []
    # Правильный обход боксов в Ultralytics
    if results and len(results) > 0:
        for box in results[0].boxes:
            # Извлекаем одномерный список координат [xmin, ymin, xmax, ymax]
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            detections.append(
                DetectionResult(
                    box=[int(x) for x in xyxy],
                    confidence=round(conf, 2),
                    label=label
                )
            )

    return PredictResponse(detections=detections)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
