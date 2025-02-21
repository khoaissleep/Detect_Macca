import cv2
import os
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

# Tạo thư mục lưu ảnh tăng cường nếu chưa có
os.makedirs("multinuts_aug", exist_ok=True)
os.makedirs("multinuts_aug/aug_images", exist_ok=True)
os.makedirs("multinuts_aug/aug_labels", exist_ok=True)

# Augmentation với cập nhật bounding box
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT_101),  # Giữ vật thể trong ảnh
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
], bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"], min_visibility=0.3))  # Giữ bounding box hợp lệ

image_dir = "multinuts/images"
label_dir = "multinuts/labels"
aug_image_dir = "multinuts_aug/aug_images"
aug_label_dir = "multinuts_aug/aug_labels"

# Xử lý từng ảnh
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi: Không thể đọc ảnh {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
        h, w, _ = image.shape

        # Đọc label
        if not os.path.exists(label_path):
            print(f"Lỗi: Không tìm thấy label {label_path}")
            continue
        
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        bboxes = []
        category_ids = []
        
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
            bboxes.append([x_center, y_center, bbox_width, bbox_height])
            category_ids.append(class_id)
        
        # Augment và lưu ảnh
        for i in range(3):
            augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]

            # Kiểm tra nếu không còn bounding box hợp lệ
            if len(aug_bboxes) == 0:
                print(f"Bỏ ảnh {filename}_aug{i} vì không có bounding box hợp lệ.")
                continue

            # Chuyển ảnh về BGR để lưu với OpenCV
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

            # Lưu ảnh tăng cường
            aug_filename = f"{filename.split('.')[0]}_aug{i}.jpg"
            aug_image_path = os.path.join(aug_image_dir, aug_filename)
            success = cv2.imwrite(aug_image_path, aug_image_bgr)

            if not success:
                print(f"Lỗi: Không thể lưu ảnh {aug_image_path}")
                continue

            print(f"Đã lưu ảnh: {aug_image_path}")

            # Lưu label tăng cường
            aug_label_path = os.path.join(aug_label_dir, aug_filename.replace(".jpg", ".txt"))
            with open(aug_label_path, "w") as f:
                for bbox, class_id in zip(aug_bboxes, category_ids):
                    f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

# Kiểm tra ảnh
img = cv2.imread(os.path.join(aug_image_dir, "image1_aug0.jpg"))
if img is None:
    print("Lỗi: Ảnh đã tăng cường không tồn tại.")
else:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Kết hợp data aug với data gốc
import os
import shutil
import hashlib

def get_file_hash(file_path):
    """Tạo hash MD5 của file để kiểm tra trùng lặp nội dung."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def merge_folders(src_folder1, src_folder2, dest_folder):
    """Hợp nhất hai thư mục mà không bị trùng file."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    existing_files = {}  # Lưu hash của file đã tồn tại
    for file in os.listdir(dest_folder):
        file_path = os.path.join(dest_folder, file)
        if os.path.isfile(file_path):
            existing_files[get_file_hash(file_path)] = file

    for src_folder in [src_folder1, src_folder2]:
        for file in os.listdir(src_folder):
            src_path = os.path.join(src_folder, file)
            if os.path.isfile(src_path):
                file_hash = get_file_hash(src_path)
                if file_hash not in existing_files:  # Nếu chưa tồn tại thì copy
                    shutil.copy2(src_path, dest_folder)
                    existing_files[file_hash] = file
                    print(f"Copied: {file}")
                else:
                    print(f"Skipped (duplicate): {file}")

# Định nghĩa đường dẫn
image_folder1 = 'multinuts/images'
image_folder2 = 'multinuts_aug/aug_images'
image_dest = 'data_train/images'

label_folder1 = 'multinuts/labels'
label_folder2 = 'multinuts_aug/aug_labels'
label_dest = 'data_train/labels'

# Gộp ảnh
print("Merging images...")
merge_folders(image_folder1, image_folder2, image_dest)

# Gộp nhãn
print("Merging labels...")
merge_folders(label_folder1, label_folder2, label_dest)

print("Merging completed!")
