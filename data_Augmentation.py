import cv2
import os
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import shutil
import hashlib

# Định nghĩa augmentation giữ ảnh rõ nét
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Lật ngang ảnh với xác suất 50%
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT_101),  # Xoay ảnh tối đa 10 độ
    A.RandomBrightnessContrast(p=0.2),  # Điều chỉnh độ sáng & độ tương phản (20%)
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),  # Làm sắc nét ảnh (30%)
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),  # Điều chỉnh màu sắc (30%)
], bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"], min_visibility=0.3))  # Giữ bounding box hợp lệ

# Định nghĩa thư mục
image_dir = "/home/khoa_is_sleep/Detect_Multinuts/DATA/datamultinuts/images"
label_dir = "/home/khoa_is_sleep/Detect_Multinuts/DATA/datamultinuts/labels"
aug_image_dir = "/home/khoa_is_sleep/Detect_Multinuts/DATA/data_aug/aug_images"
aug_label_dir = "/home/khoa_is_sleep/Detect_Multinuts/DATA/data_aug/aug_labels"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# Xử lý từng ảnh
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.rsplit(".", 1)[0] + ".txt")

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi: Không thể đọc ảnh {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB

        # Đọc label
        bboxes = []
        category_ids = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                bboxes.append(list(map(float, parts[1:])))
                category_ids.append(class_id)

        # Tạo 3 ảnh tăng cường cho mỗi ảnh gốc
        for i in range(3):
            augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]

            # Lưu ảnh tăng cường
            aug_filename = f"{filename.rsplit('.', 1)[0]}_aug{i}.jpg"
            aug_image_path = os.path.join(aug_image_dir, aug_filename)
            cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            print(f"Đã lưu ảnh: {aug_image_path}")

            # Lưu label tăng cường nếu có bounding box
            aug_label_path = os.path.join(aug_label_dir, aug_filename.replace(".jpg", ".txt"))
            with open(aug_label_path, "w") as f:
                for bbox, class_id in zip(aug_bboxes, category_ids):
                    f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

# Hợp nhất dữ liệu gốc và dữ liệu tăng cường
image_dest = "/home/khoa_is_sleep/Detect_Multinuts/DATA/data_train/images"
label_dest = "/home/khoa_is_sleep/Detect_Multinuts/DATA/data_train/labels"

os.makedirs(image_dest, exist_ok=True)
os.makedirs(label_dest, exist_ok=True)

def get_file_hash(file_path):
    """Tạo hash MD5 của file để kiểm tra trùng lặp nội dung."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def merge_folders(src_folder1, src_folder2, dest_folder):
    """Hợp nhất hai thư mục mà không bị trùng file."""
    existing_files = {}
    for file in os.listdir(dest_folder):
        file_path = os.path.join(dest_folder, file)
        if os.path.isfile(file_path):
            existing_files[get_file_hash(file_path)] = file
    
    for src_folder in [src_folder1, src_folder2]:
        for file in os.listdir(src_folder):
            src_path = os.path.join(src_folder, file)
            if os.path.isfile(src_path):
                file_hash = get_file_hash(src_path)
                if file_hash not in existing_files:
                    shutil.copy2(src_path, dest_folder)
                    existing_files[file_hash] = file
                    print(f"Copied: {file}")
                else:
                    print(f"Skipped (duplicate): {file}")

# Gộp ảnh
print("Merging images...")
merge_folders(image_dir, aug_image_dir, image_dest)

# Gộp nhãn
print("Merging labels...")
merge_folders(label_dir, aug_label_dir, label_dest)

print("Merging completed!")
