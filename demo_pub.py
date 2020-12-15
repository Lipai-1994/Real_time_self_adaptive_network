#!/usr/bin/env python3

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import roslib
import rospy
from sensor_msgs.msg import Image
import sys
import threading



class publisher(threading.Thread):
	def __init__(self, _queue, _cv):    
		self.image_pub = rospy.Publisher("camera/image", Image, queue_size=10)       
		self.bridge = CvBridge()
		self._cv = _cv
		self._queue = _queue
		threading.Thread.__init__(self)
    
	def stop(self):
		"""
		Stop the acquisition of new frames from the model
		"""
		self._stop_acquire=True

	def run(self):
		while not rospy.is_shutdown():
			try:
				with self._cv:
					# import pdb;pdb.set_trace()
					while(len(self._queue) == 0):
						print("wait")
						self._cv.wait()
					

					frame = self._queue[0]
					self._queue.pop()
					if frame is None:
						continue
					
					print("run\n")
					msg = self.bridge.cv2_to_imgmsg(frame, "mono8")
					self.image_pub.publish(msg)
					# rospy.Rate(0).sleep()
					
			
			except CvBridgeError as e:
				print(e)
		_cv.release()
	    	





