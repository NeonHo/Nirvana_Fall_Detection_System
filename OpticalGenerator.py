import numpy as np
import cv2
from threading import Lock


class OpticalGenerator:
    def __init__(self, avi_path, flow_save_path, bound, width, height, stack_length):
        self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()  # set parameters to speed up.
        self.avi_path = avi_path
        self.flow_save_path = flow_save_path
        self.bound = bound
        self.width = width
        self.height = height
        self.or_start_fe = [False]  # optical_generator get enough flow images to start feature_extractor.
        self.work_permission = False
        self.lock = Lock()
        self.stack_length = stack_length

    def generate_flow_couple_tvl1(self, frame_input_queue, flow_output_queue):
        while True:
            cv2.waitKey(1)
            # self.lock.acquire()
            previous_frame, current_frame, flow_index = frame_input_queue.get()
            # self.lock.release()
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
            flow_output_queue.put((flow[:, :, 0], flow[:, :, 1]))
            print("flow:" + str(flow_index))

    def generate_optical_flow_tvl1(self, avi_path, flow_output_queue):
        """

        :param flow_output_queue:
        :param avi_path: 要输入的视频, 包括avi视频的名称。
        :return:
        """
        video_pointer = cv2.VideoCapture(avi_path)  # get the video.
        ret, frame1 = video_pointer.read()  # read the first frame of the video.
        frame1 = cv2.resize(frame1, (self.width, self.height))
        previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # convert color BGR to gray. previous is frame1.
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        cont = 0
        while ret:
            ret, frame2 = video_pointer.read()  # read the next frame of the video.
            if not ret:
                break
            frame2 = cv2.resize(frame2, (self.width, self.height))
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # convert color BGR to gray.
            flow = self.tvl1.calc(previous_frame, next_frame, None)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow = (flow + self.bound) * (255.0 / (2 * self.bound))
            flow = np.round(flow).astype(int)
            flow[flow >= 255] = 255
            flow[flow <= 0] = 0
            hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('BGR', bgr)  # show the BGR flow image.
            k = cv2.waitKey(1) & 0xff
            if k == 27:  # esc key to escape.
                break
            flow_output_queue.put((flow[:, :, 0], flow[:, :, 1]))
            cont += 1
            previous_frame = next_frame
            print("flow:" + str(cont))

    def generate_optical_flow_farneback(self, avi_path, flow_output_queue_0):
        video_pointer = cv2.VideoCapture(avi_path)  # get the video.
        ret, frame1 = video_pointer.read()  # read the first frame of the video.
        frame1 = cv2.resize(frame1, (self.width, self.height))
        previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # convert color BGR to gray. previous is frame1.
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        count = 0
        while ret:
            ret, frame2 = video_pointer.read()  # read the next frame of the video.
            if not ret:
                break
            frame2 = cv2.resize(frame2, (self.width, self.height))
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # convert color BGR to gray.
            flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow = (flow + self.bound) * (255.0 / (2 * self.bound))
            flow = np.round(flow).astype(int)
            flow[flow >= 255] = 255
            flow[flow <= 0] = 0
            hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('BGR', bgr)  # show the BGR flow image.
            k = cv2.waitKey(1) & 0xff
            if k == 27:  # esc key to escape.
                break
            count += 1
            flow_output_queue_0.put((flow[:, :, 0], flow[:, :, 1]))
            # if count % (self.stack_length * 2) <= self.stack_length:
            #     flow_output_queue_1.put((flow[:, :, 0], flow[:, :, 1]))  # 01-10, 21-30, 41-50...
            # else:
            #     flow_output_queue_0.put((flow[:, :, 0], flow[:, :, 1]))  # 11-20, 31-40, 51-60...
            previous_frame = next_frame
            print("flow:" + str(count))
