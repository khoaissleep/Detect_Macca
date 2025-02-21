import cv2
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from ultralytics import YOLO

# Kiểm tra thiết bị (CPU do bạn đang train trên CPU)
device = "cpu"
print(f"Using device: {device}")

# Load mô hình YOLO đã huấn luyện
model_path = "/home/khoa_is_sleep/Detect_Multinuts/runs/detect/train5/weights/last.pt"
model = YOLO(model_path).to(device)

# Thư mục ảnh validation
val_images_dir = "/home/khoa_is_sleep/DETECT_macadamia-nuts-2/data_val/images"
labels_dir = "/home/khoa_is_sleep/DETECT_macadamia-nuts-2/data_val/labels"  # Nơi chứa file label

# Kiểm tra thư mục ảnh
if not os.path.exists(val_images_dir):
    print(f"❌ Lỗi: Thư mục {val_images_dir} không tồn tại!")
    exit(1)

# Chạy đánh giá trên tập validation
metrics = model.val(
    data="/home/khoa_is_sleep/DETECT_macadamia-nuts-2/data.yaml",
    batch=2,        
    imgsz=640,
    device=device,
    conf=0.001,
    iou=0.45
)

# Lấy Recall & Precision từ metrics
recalls = metrics.box.r  
precisions = metrics.box.p  

# Tìm Recall min & Precision min
recall_min = np.min(recalls) if recalls.size > 0 else None
precision_min = np.min(precisions) if precisions.size > 0 else None

# In kết quả đánh giá
print("\n📊 Evaluation Results:")
print(f"🔹 Precision: {metrics.box.p.mean():.4f}")  
print(f"🔹 Recall: {metrics.box.r.mean():.4f}")  
print(f"🔹 F1-score: {metrics.box.f1.mean():.4f}")  
print(f"🔹 mAP@0.5: {metrics.box.map50.mean():.4f}")  
print(f"🔹 mAP@0.5:0.95: {metrics.box.map.mean():.4f}")
print(f"📉 Recall min: {recall_min:.4f}" if recall_min is not None else "📉 Không có giá trị Recall min")
print(f"📉 Precision min: {precision_min:.4f}" if precision_min is not None else "📉 Không có giá trị Precision min")

# Lấy danh sách ảnh validation
image_paths = [os.path.join(val_images_dir, img) for img in os.listdir(val_images_dir) if img.endswith(('.jpg', '.png'))]

# Thống kê số ảnh không có detection và số lượng label bị bỏ qua
no_detection_count = 0
total_missed_labels = 0
total_gt_labels = 0  # Tổng số ground truth label

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))  # File label tương ứng

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Lỗi: Không thể đọc ảnh {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB

    # Đọc số lượng ground truth labels
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            gt_labels = f.readlines()
        gt_count = len(gt_labels)
    else:
        gt_count = 0

    total_gt_labels += gt_count  # Cộng dồn tổng số ground truth label

    # Dự đoán với YOLO
    results = model(img, device=device)[0]

    detected_count = len(results.boxes)
    missed_labels = max(0, gt_count - detected_count)  # Số label bị bỏ qua
    total_missed_labels += missed_labels

    if detected_count == 0:
        no_detection_count += 1  # Đếm số ảnh không có detection

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ bbox
        conf = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID

        # Vẽ bounding box của YOLO (màu xanh lá)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Pred {class_id} ({conf:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị ảnh (KHÔNG lưu)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# Tính tỷ lệ label bị bỏ qua
missed_label_percentage = (total_missed_labels / total_gt_labels * 100) if total_gt_labels > 0 else 0

# In báo cáo tổng kết
print("\n📊 Tổng kết sau khi chạy:")
print(f"🔴 Số ảnh không có detection: {no_detection_count}/{len(image_paths)}")
print(f"⚠️ Tổng số label bị bỏ qua: {total_missed_labels}/{total_gt_labels} ({missed_label_percentage:.2f}%)")

print("✅ Completed visualization of predicted labels!")
