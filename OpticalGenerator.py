import numpy as np
import cv2 as cv


class OpticalGenerator:
    def __init__(self):
        self.tvl1 = cv.DualTVL1OpticalFlow_create()
        pass

    def generate(self, avi_path, flow_save_path, bound):
        """

        :param avi_path: 要输入的视频。
        :param flow_save_path: 不含图片名称的存储路径，请设定到最底层文件夹并加斜杠。
        :return:
        """
        video_pointer = cv.VideoCapture(avi_path)  # get the video.
        ret, frame1 = video_pointer.read()  # read the first frame of the video.
        previous = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)  # convert color BGR to gray. previous is frame1.
        flow_x = np.zeros_like(frame1)  # flow_x is a zero matrix with same shape of frame1
        flow_y = np.zeros_like(frame1)  # flow_y is a zero matrix with same shape of frame1
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        cont = 0
        while ret != False:
            ret, frame2 = video_pointer.read()  # read the next frame of the video.
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  # convert color BGR to gray.
            # flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.702, 5, 10, 2, 7, 1.5, cv.OPTFLOW_FARNEBACK_GAUSSIAN)  # dense optical flow.
            flow = self.tvl1.calc(previous, next, None)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            flow = (flow + bound) * (255.0 / (2 * bound))
            flow = np.round(flow).astype(int)
            flow[flow >= 255] = 255
            flow[flow <= 0] = 0
            hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('BGR', bgr)  # show the BGR flow image.
            k = cv.waitKey(10) & 0xff
            if k == 27:  # esc key to escape.
                break
            cv.imwrite(flow_save_path + "flow_x_"+str(cont)+".jpg", flow[:, :, 0])
            cv.imwrite(flow_save_path + "flow_y_"+str(cont)+".jpg", flow[:, :, 1])
            cont += 1
            previous = next
