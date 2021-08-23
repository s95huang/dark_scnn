#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import time
from PIL import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import scipy.special
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header, Int32
from almon_msgs.msg import LanePoint, LanePoints
import cv2
import sys
import caffe

DEBUG = False # set to false when running inference

class dscnn:
    def __init__(self):
        # print(caffe.__version__)
        self.brige = CvBridge()
        caffe.set_mode_gpu() # set caffe to use GPU
        caffe.set_device(0)  # use the first gpu card  
        self.raw_img_msg = None
        # self.com_img_msg = None
        self.lane_pts_msg = None
        self.image = None
        # self.br = CvBridge()

        self.img_topic = rospy.get_param("~img_topic", '/almon/front_cam/image/compressed')
        self.lane_visual_topic = rospy.get_param("~lane_visual_topic",'/almon/lane_vis/compressed')
        self.lane_pts_topic = rospy.get_param('~lane_pts_topic','/almon/lane_pts')

        self.net = caffe.Net("/home/almon-18/Documents/dark_scnn/caffe_files/deploy.prototxt",
                             "/home/almon-18/Documents/dark_scnn/caffe_files/deploy.caffemodel",
                             caffe.TEST)
        self.net.blobs['input'].reshape(1, 3, 288, 800)

        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

        # "/almon/lane_vis/compressed"
        # /almon/lane_pts"
        self.pub_vis = rospy.Publisher(self.lane_visual_topic, CompressedImage, queue_size=1)
        self.pub_lanepts = rospy.Publisher(self.lane_pts_topic, LanePoints, queue_size=1)

        if DEBUG:
            print ('               ')
            print('DONE initialize')

        if MODEL_CULANE:
            self.row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

    def _callback_raw_img(self, raw_image):
        self.com_img_msg = raw_image




    def _listener(self):
        rospy.init_node("ultra_lane", anonymous=True)

        rospy.Subscriber(self.img_topic, CompressedImage, self._callback_raw_img, queue_size=1)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self._lane_infer()
            rate.sleep()
        # try:
        #     rospy.spin()
        # except KeyboardInterrupt:
        #     print("shutting down")





        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', result)[1]).tostring()
        # Publish new image
        self.pub_vis.publish(msg)

    def main(self):
        try:
            self._listener()
        except rospy.ROSInterruptException:
            rospy.loginfo("Shutting down node")


if __name__ == '__main__':
    node = dscnn()
    node.main()
