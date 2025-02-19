import cv2
import os
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

# Đường dẫn thư mục ảnh gốc và thư mục lưu ảnh đã tăng cường
image_dir = "mac_nuts/images"
label_dir = "mac_nuts/labels"
aug_image_dir = "mac_nuts/aug_images"
aug_label_dir = "mac_nuts/aug_labels"

# Tạo thư mục lưu ảnh tăng cường nếu chưa có
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# Định nghĩa các kỹ thuật Augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Lật ngang 50%
    A.Rotate(limit=10, p=0.5),  # Xoay ±10 độ
    A.RandomBrightnessContrast(p=0.2),  # Tăng/Giảm độ sáng
    A.GaussNoise(p=0.2),  # Thêm nhiễu Gauss
    A.RandomCrop(width=400, height=400, p=0.3),  # Cắt ngẫu nhiên
])

# Xử lý từng ảnh trong thư mục
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Chỉ lấy file ảnh
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi: Không thể đọc ảnh {image_path}")
            continue
        
        h, w, _ = image.shape

        # Đọc file label
        if not os.path.exists(label_path):
            print(f"Lỗi: Không tìm thấy label {label_path}")
            continue
        
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Tạo dữ liệu tăng cường
        for i in range(3):  # Tạo 3 ảnh augmented từ mỗi ảnh gốc
            augmented = transform(image=image)
            aug_image = augmented["image"]
            
            # Lưu ảnh tăng cường
            aug_filename = f"{filename.split('.')[0]}_aug{i}.jpg"
            aug_image_path = os.path.join(aug_image_dir, aug_filename)
            cv2.imwrite(aug_image_path, aug_image)

            # Ghi file label tương ứng
            aug_label_path = os.path.join(aug_label_dir, aug_filename.replace(".jpg", ".txt"))
            with open(aug_label_path, "w") as f:
                for line in lines:
                    f.write(line)  # Giữ nguyên label YOLO
