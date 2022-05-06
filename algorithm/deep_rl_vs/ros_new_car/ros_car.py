#!/usr/bin/env python
# -*- coding: utf-8 -*
from __future__ import print_function
import os
import sys
# from __future__ import print_function
import threading
import roslib
from rospy.topics import Publisher;roslib.load_manifest('teleop_twist_keyboard')
import rospy
from geometry_msgs.msg import Twist
import select, termios, tty
import cv2
from std_msgs.msg import String

class Roscar(threading.Thread):
    def __init__(self):
        super(Roscar, self).__init__()
        # Camera
        self.cap = cv2.VideoCapture(2)
        rospy.init_node('teleop_twist_keyboard')
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1) 
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = rospy.get_param("~speed", 0.5)
        self.turn = rospy.get_param("~turn", 1.0)
        self.condition = threading.Condition()
        self.done = False
        
        self.start()
        
    def camera_observe(self, scale = 1):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
            return frame

    def moveCar(self, moveSpeed, moveDirection):
        self.condition.acquire()
        self.condition.wait(0.5)
        self.x = moveSpeed
        self.th = moveDirection
        self.condition.notify()
        self.condition.release()
        twist = Twist()
        twist.linear.x = moveSpeed
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = moveDirection
        # self.condition.release()
        self.pub.publish(twist)

    def image_action(self,response_action):
        # response_action[0] moveSpeed response_action[1]moveDirection,range[-1,1]
        # self.moveCar(response_action[0],response_action[1])
        self.moveCar(1,0)

if  __name__ == "__main__":
    rs = Roscar()
    for i in range(10):
        rs.image_action([1,1])