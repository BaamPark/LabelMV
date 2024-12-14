import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from tools.logger_config import logger

class VideoHandler:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Cache video properties
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video_cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_video_frame(self, sequence, pixmap=True):
        if sequence < 0:
            sequence = 0
        elif sequence >= self.total_frames:
            sequence = self.total_frames - 1

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, sequence)
        ret, frame = self.video_cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if pixmap is False:
                return frame
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            return QPixmap.fromImage(image)
        return None

    def get_frame_indices(self, desired_fps):
        fps = min(desired_fps, self.fps)
        return [i for i in range(0, self.total_frames, self.fps // fps)]

    def get_video_dimensions(self):
        return self.width, self.height

    def release(self):
        self.video_cap.release()

if __name__ == '__main__':
    video_path = 'samples_vid\VP-view1.mp4'
    video_handler = VideoHandler(video_path)
    frame_indices = video_handler.get_frame_indices(1)
    print(frame_indices)
    #display 