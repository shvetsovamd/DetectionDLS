from ultralytics import YOLO, RTDETR

# --- Для YOLOv11 ---
model_yolo = YOLO("yolo11n.pt") # или yolo11s.pt / yolo11m.pt
model_yolo.train(data="dataset.yaml", epochs=50, imgsz=640, device=0)

# --- Для RT-DETR ---
model_rtdetr = RTDETR("rtdetr-l.pt")
model_rtdetr.train(data="dataset.yaml", epochs=50, imgsz=640, device=0)
