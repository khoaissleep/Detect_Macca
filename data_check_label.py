import cv2
import matplotlib.pyplot as plt
import os

# Đường dẫn đến thư mục ảnh và nhãn
image_dir = 'DATA/datanew/images'
label_dir = 'DATA/datanew/labels'

def draw_bboxes(image_path, label_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Đọc label file
    if not os.path.exists(label_path):
        print(f"Lỗi: Không tìm thấy label {label_path}")
        return
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
        
        # Chuyển từ YOLO sang pixel
        x1 = int((x_center - bbox_width / 2) * w)
        y1 = int((y_center - bbox_height / 2) * h)
        x2 = int((x_center + bbox_width / 2) * w)
        y2 = int((y_center + bbox_height / 2) * h)
        
        # Vẽ hình chữ nhật
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, "mac", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Hiển thị ảnh
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# Duyệt tất cả file trong thư mục ảnh
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Chỉ lấy file ảnh
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))
        
        print(f"Kiểm tra ảnh: {image_path}")
        draw_bboxes(image_path, label_path)

