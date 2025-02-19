import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load mô hình đã huấn luyện
model = YOLO("runs/detect/train/weights/best.pt")  # Đường dẫn tới model YOLOv8 tốt nhất

# Đánh giá trên tập validation
metrics = model.val()

# In ra các chỉ số đánh giá
print(f"mAP50: {metrics.box.map50:.4f}")  # Mean Average Precision (mAP) @ IoU 0.5
print(f"mAP50-95: {metrics.box.map:.4f}")  # mAP @ IoU 0.5:0.95
print(f"Precision: {metrics.box.precision:.4f}")
print(f"Recall: {metrics.box.recall:.4f}")

# Thư mục chứa ảnh kiểm tra
val_dir = "mac_nuts/images"  # Đổi sang tập test nếu có
output_dir = "mac_nuts/results"
os.makedirs(output_dir, exist_ok=True)

# Chạy dự đoán trên từng ảnh trong thư mục kiểm tra
for filename in os.listdir(val_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(val_dir, filename)
        
        # Dự đoán ảnh
        results = model(image_path)
        
        # Vẽ bounding box trên ảnh
        for result in results:
            img = result.plot()  # Vẽ bounding box trên ảnh
        
            # Lưu ảnh kết quả
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)

            # Hiển thị ảnh
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
            print(f"Đã lưu ảnh kết quả tại: {output_path}")

print("Hoàn tất đánh giá!")
