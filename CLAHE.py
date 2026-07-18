import cv2
import os
from glob import glob


def apply_clahe_to_dataset(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Рентген обычно одноканальный
        enhanced = clahe.apply(img)
        # Переводим обратно в 3-канальный (RGB), так как YOLO/RT-DETR ждут 3 канала
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), enhanced_rgb)
