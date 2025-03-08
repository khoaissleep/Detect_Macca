import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Load mô hình SAM
sam_checkpoint = "sam_vit_b.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to("cpu")
sam_predictor = SamPredictor(sam)

# Định nghĩa khoảng màu nâu trong HSV
LOWER_BROWN = np.array([10, 50, 20])
UPPER_BROWN = np.array([30, 255, 200])

def is_brown(image, bbox):
    """ Kiểm tra xem vật thể trong bbox có màu nâu hay không """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    
    if roi is None or roi.size == 0:
        return False  # Tránh lỗi khi bbox nằm ngoài ảnh
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, LOWER_BROWN, UPPER_BROWN)
    brown_ratio = np.sum(mask > 0) / mask.size
    return brown_ratio > 0.5

class CustomYOLO(YOLO):
    """ Tùy chỉnh YOLO để ưu tiên học vật thể màu nâu """
    def custom_loss(self, predictions, targets, images):
        total_loss = 0
        for i in range(len(images)):
            img = images[i]
            target = targets[i]
            bbox = target["bboxes"]
            
            if is_brown(img, bbox):
                loss_weight = 2.0  # Tăng trọng số nếu là màu nâu
            else:
                loss_weight = 1.0
            
            loss = super().compute_loss(predictions, targets) * loss_weight
            total_loss += loss
        return total_loss

# Load model YOLO
model = CustomYOLO("yolov8n.pt")

def segment_with_sam(image, bbox):
    """ Sử dụng SAM để tạo mask từ bbox của YOLO """
    sam_predictor.set_image(image)
    x1, y1, x2, y2 = map(int, bbox)
    input_box = np.array([x1, y1, x2, y2])
    masks, _, _ = sam_predictor.predict(box=input_box, multimask_output=False)
    return masks[0]  # Trả về mask đầu tiên

# Kiểm tra dữ liệu đầu vào
image_path = "/home/khoa_is_sleep/Detect_Multinuts/DATA/data_train/images/anh1_aug0.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Không thể đọc ảnh {image_path}")

# Train model
model.train(
    data="data.yaml",
    epochs=6,
    batch=2,
    imgsz=640,
    device="cpu",
    mosaic=0,
    mixup=0,
    cache=True,
    workers=0
)