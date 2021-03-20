import cv2


class Videographer:
    def __init__(self, video_path, width, height):
        self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_path = video_path
        self.width = width
        self.height = height
        self.out = cv2.VideoWriter(self.video_path + "video.avi", self.fourcc, 20.0, (self.width, self.height))
        self.previous_frame = None
        self.captured_frame_number = 0

    def capture_video_frame(self):
        """
        capture 1 RGB frame, show it and store into the video.
        :return:
        """
        ret, current_frame = self.capture.read()
        if current_frame is None:
            print("您在USB接口1上可能没有接摄像头！并未捕获到录像帧！")
        self.captured_frame_number += 1
        current_frame = cv2.resize(current_frame, (self.width, self.height))
        cv2.imshow("Camera", current_frame)
        self.out.write(current_frame)
        return ret, current_frame

    def capture_video(self, frame_output_queue):
        ret = True
        while ret:
            ret, current_frame = self.capture_video_frame()
            if self.captured_frame_number <= 1:
                self.previous_frame = current_frame
                continue  # the previous frame is None at the 1st time.
            else:
                frame_output_queue.put((self.previous_frame, current_frame, self.captured_frame_number))
                # print("拍摄第-"+str(self.captured_frame_number)+"张图像")
                self.previous_frame = current_frame
                key = cv2.waitKey(1) & 0xff
                if key == 27:  # esc key to escape.
                    break
