from ultralytics import YOLO
import torch
from torchvision.ops import box_iou

# Load model
model = YOLO("/home/khoa_is_sleep/DETECT_macadamia-nuts-1/best.pt")

# Đọc dữ liệu test
import os
image_dir = "mac_nuts/images"
test_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]


# Khởi tạo bộ đếm
TP, FP, FN = 0, 0, 0

for image_path in test_images:
    results = model(image_path)  # Dự đoán trên ảnh

    # Lấy bounding boxes dự đoán
    preds = results[0].boxes.xyxy.cpu()  # Dạng (x1, y1, x2, y2)
    pred_classes = results[0].boxes.cls.cpu()  # Lấy class

    # Đọc ground truth từ file label YOLO
    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    with open(label_path, "r") as f:
        gt_boxes = []
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

            # Chuyển từ YOLO format -> (x1, y1, x2, y2)
            img_w, img_h = 640, 640  # Giả sử ảnh 640x640, hoặc lấy kích thước thật
            x1 = int((x_center - bbox_width / 2) * img_w)
            y1 = int((y_center - bbox_height / 2) * img_h)
            x2 = int((x_center + bbox_width / 2) * img_w)
            y2 = int((y_center + bbox_height / 2) * img_h)
            gt_boxes.append([x1, y1, x2, y2])

    gt_boxes = torch.tensor(gt_boxes)  # Ground truth boxes

    # Tính IoU giữa dự đoán và ground truth
    if len(preds) > 0 and len(gt_boxes) > 0:
        iou = box_iou(preds, gt_boxes)
        max_iou, _ = iou.max(dim=1)

        TP += (max_iou > 0.5).sum().item()  # TP: dự đoán đúng (IoU > 0.5)
        FP += (max_iou <= 0.5).sum().item()  # FP: Dự đoán sai
        FN += (len(gt_boxes) - (max_iou > 0.5).sum().item())  # FN: Bỏ sót GT
    else:
        FP += len(preds)
        FN += len(gt_boxes)

# Tính Precision, Recall, F1-score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
