Детекция переломов на рентгеновских снимках

Данные

https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project


Модели

- YOLOv11 - baseline
- RT-DETR - точнее

mAP - метрика

Хакинг mAP на датасете

1. Предобработка (Image Enhancement): Рентгеновские снимки часто страдают от низкого контраста.
Я применяю CLAHE (Contrast Limited Adaptive Histogram Equalization) перед подачей снимков в модель
2. Разрешение изображения: Переломы — мелкие объекты. Модель обучена на разрешении не менее 640x640,
так как сильное сжатие сотрет тонкие линии переломов

- Backend - FastAPI
- Frontend - Streamlit

Запуск

- python app_backend.py
- streamlit run app_frontend.py
