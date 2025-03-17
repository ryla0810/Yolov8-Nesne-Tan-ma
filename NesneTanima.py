import cv2
from ultralytics import YOLO

# Eğitilmiş modeli yükleyin
model = YOLO('runs/detect/train/weights/best.pt')

# Videoyu aç
cap = cv2.VideoCapture('col.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Her karede tespit yapın
    results = model(frame)

    # Tespit edilen nesneleri kutularla işaretleyip kare üzerine çizin
    annotated_frame = results[0].plot()

    # Sonuçları göster
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
