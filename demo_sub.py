#!/usr/bin/env python3

import cv2 as cv
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys

class Subscriber():
	def __init__(self):	    
		self.bridge = CvBridge()

	def callback(self, msg):
		try:
			# import pdb; pdb.set_trace()
			bridge = CvBridge()

			# Only compatiable with python2 currently
			cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")

			cv.imshow("ROS_Sub", cv_image)
			cv.waitKey(1)
		except CvBridgeError as e:
			print(e)

def main():
    # import pdb; pdb.set_trace()
	sub = Subscriber();
	rospy.init_node('Image_Sub', anonymous=True)
	rospy.Subscriber("camera/image", Image, sub.callback)
	rospy.spin()
	
	# cv.destroyAllWindows()

if __name__ == '__main__':
	main()