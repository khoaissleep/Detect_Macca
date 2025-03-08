"""   Các hàm chính:
- main(): Khởi chạy hệ thống theo dõi
- get_brown_ratio(): Tính tỷ lệ màu nâu trong vùng quan tâm
- analyze_object(): Phân tích và kiểm tra tính hợp lệ của hạt
- update_tracks(): Cập nhật thông tin theo dõi hạt
- _assign_id(): Gán ID cho hạt mới
- update_display(): Hiển thị kết quả trực quan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# Cấu hình logging để theo dõi quá trình xử lý
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Lớp cấu hình chứa tất cả các tham số của hệ thống
    Sử dụng dataclass để quản lý các tham số một cách có tổ chức
    """
    # Tham số phát hiện màu nâu trong không gian HSV
    LOWER_BROWN: np.ndarray = field(default_factory=lambda: np.array([10, 50, 20]))  # Giới hạn dưới
    UPPER_BROWN: np.ndarray = field(default_factory=lambda: np.array([30, 255, 200]))  # Giới hạn trên
    
    # Các ngưỡng phát hiện
    CONF_THRESHOLD: float = 0.3    # Ngưỡng độ tin cậy tối thiểu cho YOLO
    BROWN_THRESHOLD: float = 0.1   # Ngưỡng tỷ lệ màu nâu tối thiểu
    SIZE_MIN: int = 0             # Kích thước nhỏ nhất của hạt
    SIZE_MAX: int = 1000          # Kích thước lớn nhất của hạt
    HW_DIFF_THRESHOLD: float = 0.04  # Ngưỡng chênh lệch tỷ lệ chiều cao/chiều rộng để phân biệt hạt tròn/méo
    
    # Tham số theo dõi
    TRACKING_DISTANCE_THRESHOLD: float = 50.0  # Khoảng cách tối đa giữa các vị trí của cùng một hạt

class ObjectDetector:
    """
    Lớp phát hiện và phân tích hạt điều trong ảnh
    Sử dụng YOLOv8 để phát hiện và phân tích các thuộc tính của hạt
    """
    
    def __init__(self, model_path: str, config: Config):
        """
        Khởi tạo detector
        
        Args:
            model_path: Đường dẫn đến file trọng số mô hình YOLOv8
            config: Đối tượng cấu hình chứa các tham số
        """
        self.model = YOLO(model_path)
        self.config = config
        logger.info(f"Đã khởi tạo ObjectDetector với mô hình: {model_path}")

    def get_brown_ratio(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, np.ndarray]:
        """Tính tỷ lệ pixel màu nâu trong vùng bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, self.config.LOWER_BROWN, self.config.UPPER_BROWN)
        return np.sum(mask > 0) / mask.size, mask

    def analyze_object(self, box, frame: np.ndarray) -> Optional[Tuple]:
        """Phân tích và kiểm tra tính hợp lệ của hạt"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        
        if conf < self.config.CONF_THRESHOLD:
            logger.debug(f"Độ tin cậy thấp: {conf:.2f}")
            return None
        
        brown_ratio, _ = self.get_brown_ratio(frame, (x1, y1, x2, y2))
        if brown_ratio < self.config.BROWN_THRESHOLD:
            logger.debug(f"Tỷ lệ màu nâu thấp: {brown_ratio:.2f}")
            return None

        width, height = x2 - x1, y2 - y1
        if not (self.config.SIZE_MIN <= width <= self.config.SIZE_MAX and 
                self.config.SIZE_MIN <= height <= self.config.SIZE_MAX):
            logger.debug(f"Kích thước không hợp lệ: {width}x{height}")
            return None

        cx_bbox, cy_bbox = (x1 + x2) // 2, (y1 + y2) // 2
        major_axis = max(width, height) // 2
        minor_axis = min(width, height) // 2
        ellipse_area = math.pi * major_axis * minor_axis
        
        hw_ratio = width / height
        shape_type = "Tron" if abs(hw_ratio - 1) <= self.config.HW_DIFF_THRESHOLD else "Meo"
        
        # Phân loại kích thước dựa trên diện tích
        size_type = "To" if ellipse_area > 3400 else "Nho"
        
        return (cx_bbox, cy_bbox, major_axis, minor_axis, ellipse_area, brown_ratio, shape_type, hw_ratio, conf, size_type)

class ObjectTracker:
    """
    Lớp theo dõi hạt điều qua các frame
    Sử dụng thuật toán matching dựa trên khoảng cách để duy trì ID của hạt
    """
    
    def __init__(self, config: Config):
        """
        Khởi tạo tracker
        
        Args:
            config: Đối tượng cấu hình
        """
        self.config = config
        self.object_id_counter = 0  # Bộ đếm ID tự động tăng
        self.tracked_objects = {}    # Lưu trữ thông tin hạt đang theo dõi
        logger.info("Đã khởi tạo ObjectTracker")

    def update_tracks(self, detected_objects: List[Tuple]) -> Dict:
        """Cập nhật thông tin theo dõi với các hạt mới phát hiện"""
        new_tracked_objects = {}
        
        for obj in detected_objects:
            cx_bbox, cy_bbox, _, _, _, _, shape_type, _, _, size_type = obj
            assigned_id = self._assign_id(cx_bbox, cy_bbox)
            new_tracked_objects[assigned_id] = (cx_bbox, cy_bbox, shape_type, size_type)
            
        self.tracked_objects = new_tracked_objects
        return new_tracked_objects

    def _assign_id(self, cx: int, cy: int) -> int:
        """Gán ID cho hạt mới dựa trên khoảng cách với các hạt đã theo dõi"""
        min_distance = float("inf")
        assigned_id = None

        for obj_id, (prev_cx, prev_cy, _, _) in self.tracked_objects.items():
            distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            if distance < min_distance and distance < self.config.TRACKING_DISTANCE_THRESHOLD:
                min_distance = distance
                assigned_id = obj_id
        
        if assigned_id is None:
            self.object_id_counter += 1
            assigned_id = self.object_id_counter
            
        return assigned_id

class Visualizer:
    """
    Lớp hiển thị kết quả phát hiện và theo dõi
    Vẽ các hình elip và thông tin lên frame
    """
    
    def __init__(self):
        """Khởi tạo visualizer với matplotlib"""
        plt.ion()  # Bật chế độ tương tác
        self.fig, self.ax = plt.subplots()
        logger.info("Đã khởi tạo Visualizer")

    def update_display(self, frame: np.ndarray, tracked_objects: Dict, 
                      detected_objects: List[Tuple]) -> None:
        """Hiển thị kết quả phát hiện và theo dõi lên frame"""
        for obj_id, (cx_bbox, cy_bbox, shape_type, size_type) in tracked_objects.items():
            for det in detected_objects:
                if det[0] == cx_bbox and det[1] == cy_bbox:
                    _, _, major_axis, minor_axis, ellipse_area, brown_ratio, _, hw_ratio, conf, _ = det
                    break
            
            # Vẽ ellipse và thông tin
            cv2.ellipse(frame, (cx_bbox, cy_bbox), (major_axis, minor_axis), 
                       0, 0, 360, (0, 255, 255), 2)
            
            label = f"ID {obj_id} ({cx_bbox}, {cy_bbox}) {size_type} {shape_type}"
            cv2.putText(frame, label, (cx_bbox - 30, cy_bbox - major_axis - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            logger.info(f"[ID {obj_id}] Tâm: ({cx_bbox}, {cy_bbox}), "
                       f"Trục lớn: {major_axis}, Trục nhỏ: {minor_axis}, "
                       f"Diện tích: {ellipse_area:.2f} px², Màu nâu: {brown_ratio*100:.2f}%, "
                       f"Độ tin cậy: {conf:.2f}, Hình dạng: {shape_type}, "
                       f"Kích thước: {size_type}, "
                       f"Chênh lệch h/w-1: {abs(hw_ratio - 1):.2f}")

        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.imshow(annotated_frame)
        self.ax.axis("off")
        plt.pause(0.01)

def main():
    """Khởi chạy hệ thống theo dõi hạt điều"""
    config = Config()
    detector = ObjectDetector("best_model_9.pt", config)
    tracker = ObjectTracker(config)
    visualizer = Visualizer()
    
    video_path = "/home/khoa_is_sleep/Detect_Multinuts/test_best.mp4"
    cap = cv2.VideoCapture(video_path)          
    if not cap.isOpened():
        logger.error(f"Không thể mở video: {video_path}")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = detector.model(frame)[0]
            detected_objects = []
            
            for box in results.boxes:
                obj_info = detector.analyze_object(box, frame)
                if obj_info:
                    detected_objects.append(obj_info)
            
            tracked_objects = tracker.update_tracks(detected_objects)
            visualizer.update_display(frame, tracked_objects, detected_objects)
            
    except Exception as e:
        logger.error(f"Lỗi trong vòng lặp chính: {str(e)}")
    finally:
        cap.release()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()  