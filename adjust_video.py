import cv2
from PyQt5.QtGui import QImage, QPixmap
from logger_config import logger

def get_frame_indices(video_path, fps):
    video_cap = cv2.VideoCapture(video_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the Framerate
    original_fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    if fps > original_fps:
        fps = original_fps

    video_cap.release()
    frame_indices = [i for i in range(0, total_frames, original_fps // fps)]
    return frame_indices


def get_video_frame(video_path, sequence):
    video_cap = cv2.VideoCapture(video_path)

    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if sequence < 0:
        logger.info(f'The sequence number is negative, {sequence}. (adjust_video.py)')
        sequence = 0
    
    elif sequence > total_frames:
        logger.info(f'The sequence number is greater than the total number of video frames, {sequence}. (adjust_video.py)')
        sequence = total_frames - 1

    video_cap.set(cv2.CAP_PROP_POS_FRAMES, sequence)
    ret, frame = video_cap.read()
    video_cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        return pixmap
    
    
def get_video_dimensions(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    cap.release()
    
    return width, height
        
# if __name__ == '__main__':
#     video_path = 'samples_vid/200332_above_trimed.avi'
#     total_frames = get_frame_indices(video_path, 10)
#     print(total_frames)
    # frame = get_video_frame(video_path, 100)
    # if frame is not None:
    #     # Do something with the frame, e.g., display it
    #     cv2.imshow('Frame', frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()