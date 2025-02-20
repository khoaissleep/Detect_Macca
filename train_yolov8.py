from ultralytics import YOLO

# Đường dẫn file cấu hình dữ liệu YOLO
data_yaml = "/home/khoa_is_sleep/DETECT_macadamia-nuts-2/data.yaml"  # Đảm bảo file này tồn tại

# Tải mô hình YOLOv8 pre-trained
model = YOLO("yolov8n.pt")  # Có thể thay bằng yolov8s.pt, yolov8m.pt nếu muốn mô hình lớn hơn

# Huấn luyện mô hình
model.train(
    data=data_yaml,   # Đường dẫn đến file .yaml chứa thông tin dữ liệu
    epochs=70,        # Số epoch (có thể tăng nếu dataset lớn)
    batch=10,         # Batch size (tùy thuộc vào GPU)
    imgsz=640,        # Kích thước ảnh
    device="cuda"     # Dùng GPU (hoặc "cpu" nếu không có GPU)
)

# Lưu mô hình đã huấn luyện
model.export(format="onnx")  # Xuất mô hình sang định dạng ONNX nếu cần
