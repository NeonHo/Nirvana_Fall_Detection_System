import cv2


class Videographer:
    def __init__(self):
        self.capture1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

    def get_video_constantly(self, video_path, width, height):
        """

        :param video_path: 保存录像的文件夹路径，不包含录像名称。
        :param width: 录像帧的宽度
        :param height: 录像帧的高度
        :return:
        """
        out = cv2.VideoWriter(video_path + "video.avi", self.fourcc, 20.0, (width, height))
        ret1 = True
        while ret1:
            ret1, frame1 = self.capture1.read()
            frame1 = cv2.resize(frame1, (width, height))
            cv2.imshow("Camera", frame1)
            k = cv2.waitKey(1) & 0xff
            if k == 27:  # esc key to escape.
                break
            out.write(frame1)
