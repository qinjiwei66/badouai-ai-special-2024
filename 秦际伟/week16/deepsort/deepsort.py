import cv2
import torch 
from ultralytics import YOLO


# 加载YOLO模型用于目标检测
yolo_model = YOLO('yolov5s.pt')
yolo_model.eval()


# 初始化DeepSort跟踪器
class DeepSortTracker:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        confirmed_tracks = []
        for det in detections:
            matched = False
            for i, trk in enumerate(self.trackers):
                center_det = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
                center_trk = [(trk[0] + trk[2]) / 2, (trk[1] + trk[3]) / 2]
                dist = ((center_det[0] - center_trk[0]) ** 2 + (center_det[1] - center_trk[1]) ** 2) ** 0.5

                if dist < 50:  # 假设距离小于50像素为匹配
                    self.trackers[i] = det
                    confirmed_tracks.append(det)
                    matched = True
                    break

            if not matched:
                self.trackers.append(det)

        return confirmed_tracks


# 打开视频文件
cap = cv2.VideoCapture('video.mp4')
tracker = DeepSortTracker()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型进行目标检测
    results = yolo_model(frame)

    # 提取检测结果
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        if conf > 0.5:  # 假设置信度大于0.5为目标
            detections.append([x1, y1, x2 - x1, y2 - y1])

    # 使用DeepSort跟踪器进行目标跟踪
    confirmed_tracks = tracker.update(detections)

    # 在帧上绘制跟踪结果
    for track in confirmed_tracks:
        x1, y1, w, h = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)

    cv2.inshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   


cap.release()
cv2.destroyAllWindows()