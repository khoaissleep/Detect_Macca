import os
import shutil
import random
from pathlib import Path

# Định nghĩa đường dẫn thư mục
data_train = Path("data_train")  # Thư mục chứa ảnh & nhãn gốc
data_val = Path("data_val")  # Thư mục chứa dữ liệu validation

# Tạo thư mục validation nếu chưa có
(data_val / "images").mkdir(parents=True, exist_ok=True)
(data_val / "labels").mkdir(parents=True, exist_ok=True)

# Lấy danh sách tất cả ảnh trong thư mục train
list_imgs_path = list((data_train / "images").glob("*.jpg"))  # Hoặc "*.png" nếu dùng PNG

# Chia 20% dữ liệu cho validation
len_val_data = len(list_imgs_path) // 5
list_imgs_path_val = random.sample(list_imgs_path, len_val_data)

# Di chuyển ảnh và nhãn tương ứng sang thư mục validation
for img_path in list_imgs_path_val:
    label_path = data_train / "labels" / f"{img_path.stem}.txt"

    shutil.move(img_path, data_val / "images" / img_path.name)  # Di chuyển ảnh
    if label_path.exists():
        shutil.move(label_path, data_val / "labels" / label_path.name)  # Di chuyển nhãn nếu có

print(f"✅ Đã di chuyển {len(list_imgs_path_val)} ảnh và nhãn sang thư mục validation.")
