import asyncio
import base64
import json
import logging
import sys
import threading
import time

from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaRecorder
import requests

import cv2
import numpy as np
from datetime import datetime

from bosdyn.client.command_line import (Command, Subcommands)
from .webrtc_client import WebRTCClient

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import Trigger, TriggerResponse

class WebRTCROS():

    def __init__(self):
        pass

    # WebRTC must be in its own thread with its own event loop.
    def start_webrtc(self, shutdown_flag, process_func, recorder=None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        asyncio.gather(self.webrtc_client.start(), process_func(shutdown_flag),
                       self.monitor_shutdown(shutdown_flag))
        loop.run_forever()

    # Frame processing occurs; otherwise it waits.
    async def process_frame(self, shutdown_flag):
        while asyncio.get_event_loop().is_running():
            try:
                frame = await self.webrtc_client.video_frame_queue.get()
                self.pil_image = frame.to_image()
                # cv_image = np.array(self.pil_image) 
                # cv2.imshow('display', cv_image)
                # cv2.waitKey(1)

                self.cam_image.header.stamp = rospy.Time.now()
                self.cam_image.data = np.array(self.pil_image).tobytes()
                self.spot_cam_image_pub.publish(self.cam_image)

                # params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                # cv_image = cv2.cvtColor(np.array(self.pil_image), cv2.COLOR_BGR2RGB) 

                # self.cam_compresed_image.header.stamp = rospy.Time.now()
                # self.cam_compresed_image.data = np.array(cv2.imencode(".jpg", cv_image, params)[1]).tostring()
                # self.spot_cam_compressed_image_pub.publish(self.cam_compresed_image)
                continue
            except:
                pass
        shutdown_flag.set()

    # Flag must be monitored in a different coroutine and sleep to allow frame
    # processing to occur.
    async def monitor_shutdown(self, shutdown_flag):
        while not shutdown_flag.is_set():
            await asyncio.sleep(1.0)

        await self.webrtc_client.pc.close()
        asyncio.get_event_loop().stop()

    def save_image(self, req):
        resp = TriggerResponse()
        try:
            cv_image = cv2.cvtColor(np.array(self.pil_image) , cv2.COLOR_BGR2RGB) 
            cv2.imwrite(str(datetime.now()) + '.jpg', cv_image)
            resp.success = True
            resp.message = 'Saved as ' + str(datetime.now()) + '.jpg'
        except:
            resp.success = False
        return resp

    def main(self):
        """Main function for the WebRTC class"""
        rospy.init_node('webrtc_ros', anonymous=True)

        self.username = rospy.get_param('~username', 'default_value')
        self.password = rospy.get_param('~password', 'default_value')
        self.hostname = rospy.get_param('~hostname', 'default_value')

        self.spot_cam_image_pub = rospy.Publisher('spot_cam/image', Image, queue_size=10)
        self.spot_cam_compressed_image_pub = rospy.Publisher('spot_cam/image/compressed', CompressedImage, queue_size=10)
        rospy.Service("spot_cam/save", Trigger, self.save_image)

        self.pil_image = None
        self.cam_image = Image()
        self.cam_image.header.frame_id = 'spot_cam'
        self.cam_image.height = 720
        self.cam_image.width = 1280
        self.cam_image.encoding = 'rgb8'
        self.cam_image.is_bigendian = False
        self.cam_image.step = self.cam_image.width * 3
        self.cam_compresed_image = CompressedImage()
        self.cam_compresed_image.header.frame_id = 'spot_cam'
        self.cam_compresed_image.format = "jpeg"

        self.config = RTCConfiguration(iceServers=[])
        self.webrtc_client = WebRTCClient(self.hostname, self.username, self.password, 31102,
                                          'h264.sdp', False, self.config, media_recorder=None)
        self.shutdown_flag = threading.Event()
        self.webrtc_thread = threading.Thread(target=self.start_webrtc,
                                              args=[self.shutdown_flag, self.process_frame], 
                                              daemon=True)
        self.webrtc_thread.start()
        
        while not rospy.is_shutdown():
            try:
                self.webrtc_thread.join()
            except KeyboardInterrupt:
                self.shutdown_flag.set()