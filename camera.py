import cv2


class VideoRecorder():
    def __init__(self):
        self.video_cam = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video_cam.release()
    
    def get_frame(self): 
        ret, frame = self.video_cam.read()
        if not ret:
            return []
        else:
            # print("success!")
            return frame
