import cv2
import os
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

# Tạo thư mục lưu ảnh tăng cường nếu chưa có
os.makedirs("mac_nuts/aug_images", exist_ok=True)
os.makedirs("mac_nuts/aug_labels", exist_ok=True)

# Augmentation với cập nhật bounding box
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.RandomCrop(width=400, height=400, p=0.3),
], bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]))

image_dir = "mac_nuts/images"
label_dir = "mac_nuts/labels"
aug_image_dir = "mac_nuts/aug_images"
aug_label_dir = "mac_nuts/aug_labels"

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
