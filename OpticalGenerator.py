import threading
import numpy as np
import cv2 as cv


class OpticalGenerator(threading.Thread):
    def __init__(self, avi_path, flow_save_path, bound, width, height):
        super().__init__(name="optical_flow_generator_thread")
        self.tvl1 = cv.optflow.DualTVL1OpticalFlow_create()
        self.avi_path = avi_path
        self.flow_save_path = flow_save_path
        self.bound = bound
        self.width = width
        self.height = height
        self.or_start_fe = [False]  # optical_generator get enough flow images to start feature_extractor.

    def generate(self):
        video_pointer = cv.VideoCapture(self.avi_path + "video.avi")  # get the video.
        ret, frame1 = video_pointer.read()  # read the first frame of the video.
        # frame1 = cv.resize(frame1, (self.width, self.height))
        previous_frame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)  # convert color BGR to gray. previous is frame1.
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        count = 0
        while ret:
            ret, frame2 = video_pointer.read()  # read the next frame of the video.
            if not ret:
                break
            # frame2 = cv.resize(frame2, (self.width, self.height))
            next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  # convert color BGR to gray.
            flow = self.tvl1.calc(previous_frame, next_frame, None)  # TVL-1 is better with changing lighting conditions
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            flow = (flow + self.bound) * (255.0 / (2 * self.bound))
            flow = np.round(flow).astype(int)
            flow[flow >= 255] = 255
            flow[flow <= 0] = 0
            hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('BGR', bgr)  # show the BGR flow image.
            k = cv.waitKey(1) & 0xff
            if k == 27:  # esc key to escape.
                break
            cv.imwrite(self.flow_save_path + "flow_x_" + str(count) + ".jpg", flow[:, :, 0])
            cv.imwrite(self.flow_save_path + "flow_y_" + str(count) + ".jpg", flow[:, :, 1])
            if count == 10:
                self.or_start_fe[0] = True
            count += 1
            previous_frame = next_frame

    def run(self) -> None:
        self.generate()
