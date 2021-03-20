import numpy as np
import cv2
from threading import Lock


class OpticalGenerator:
    def __init__(self, avi_path, flow_save_path, bound, width, height):
        self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1, epsilon=0.05, warps=1)  # set parameters to speed up.
        self.avi_path = avi_path
        self.flow_save_path = flow_save_path
        self.bound = bound
        self.width = width
        self.height = height
        self.or_start_fe = [False]  # optical_generator get enough flow images to start feature_extractor.
        self.work_permission = False
        self.lock = Lock()

    def generate_flow_couple(self, frame_input_queue, flow_output_queue):
        while True:
            cv2.waitKey(1)
            # self.lock.acquire()
            previous_frame, current_frame, flow_index = frame_input_queue.get()
            # self.lock.release()
            # print("加工第-" + str(flow_index) + "张RGB图像")
            hsv = np.zeros_like(previous_frame)
            previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            flow = self.tvl1.calc(previous_frame, current_frame,
                                  None)  # TVL-1 is better with changing lighting conditions
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow = (flow + self.bound) * (255.0 / (2 * self.bound))
            flow = np.round(flow).astype(int)
            flow[flow >= 255] = 255
            flow[flow <= 0] = 0
            hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('flow_BGR', bgr)  # show the BGR flow image.
            # cv2.imwrite(self.flow_save_path + "flow_x_" + str(flow_index) + ".jpg", flow[:, :, 0])
            # cv2.imwrite(self.flow_save_path + "flow_y_" + str(flow_index) + ".jpg", flow[:, :, 1])
            flow_output_queue.put((flow[:, :, 0], flow[:, :, 1]))
            # print("加工第-" + str(flow_index) + "张RGB图像完成")
