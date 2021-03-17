import cv2
import threading


class Videographer(threading.Thread):
    def __init__(self, video_path, width, height):
        super().__init__(name="videographer_thread")
        self.capture1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_path = video_path
        self.width = width
        self.height = height
        self.v_start_og = [False]  # videographer get enough frames to start optical generator.

    def get_video_constantly(self):
        out = cv2.VideoWriter(self.video_path + "video.avi", self.fourcc, 20.0, (self.width, self.height))
        ret1 = True
        count = 0
        while ret1:
            ret1, frame_current = self.capture1.read()
            if frame_current is None:
                print("您在USB接口1上可能没有接摄像头！并未捕获到录像帧！")
            count += 1
            frame_current = cv2.resize(frame_current, (self.width, self.height))
            cv2.imshow("Camera", frame_current)
            out.write(frame_current)
            frame_previous = frame_current
            if count == 10:
                self.v_start_og[0] = True
            k = cv2.waitKey(1) & 0xff
            if k == 27:  # esc key to escape.
                break

    def run(self) -> None:
        self.get_video_constantly()
