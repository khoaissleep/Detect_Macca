from ultralytics import YOLO
import torch

# Kiểm tra xem có GPU không, nếu không thì dùng CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load mô hình đã huấn luyện (best.pt)
model_path = "/home/khoa_is_sleep/DETECT_macadamia-nuts-2/runs/detect/train2/weights/best.pt"
model = YOLO(model_path).to(device)

# Đánh giá mô hình trên tập validation
metrics = model.val(
    data="/home/khoa_is_sleep/DETECT_macadamia-nuts-2/data.yaml",  # File cấu hình dataset
    batch=10,       # Batch size (có thể điều chỉnh)
    imgsz=640,      # Kích thước ảnh
    device=device,  # Chạy trên GPU hoặc CPU
    conf=0.001,     # Ngưỡng confidence thấp để tính mAP chính xác
    iou=0.5         # IoU threshold
)

# In kết quả đánh giá
print("\n📊 Evaluation Results:")
print(f"Precision: {metrics.box.map50:.4f}")    # mAP@0.5
print(f"Recall: {metrics.box.map75:.4f}")       # Recall @ 0.75 IoU
print(f"mAP@0.5: {metrics.box.map50:.4f}")      # mAP@0.5
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")   # mAP@0.5:0.95
