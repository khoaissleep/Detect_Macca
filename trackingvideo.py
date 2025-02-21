import cv2
from ultralytics import YOLO

# Load mô hình YOLOv8 đã train
model = YOLO("/home/khoa_is_sleep/Detect_Multinuts/runs/detect/train5/weights/best.pt")

# Kết nối với camera (0 là camera mặc định)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Sử dụng Video4Linux2


# Kiểm tra camera có mở được không
if not cap.isOpened():
    print("Lỗi: Không thể mở camera!")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Nếu không lấy được frame thì thoát

    # Chạy YOLO trên frame từ camera
    results = model(frame)

    # Vẽ kết quả lên frame
    annotated_frame = results[0].plot()

    # Hiển thị video real-time
    cv2.imshow("YOLOv8 - Camera Detection", annotated_frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
