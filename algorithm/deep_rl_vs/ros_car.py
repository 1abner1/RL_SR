#!/usr/bin/env python  
# -*- coding: utf-8 -*    
import os  
import sys  
# import tty, termios  
import roslib; roslib.load_manifest('smartcar_teleop')  
import rospy  
import cv2
from geometry_msgs.msg import Twist  
from std_msgs.msg import String  
  


class Roscar:
    def __init__(self):
        #Camera 
        self.cap = cv2.VideoCapture(0)
        self.cmd =  Twist()
        self.pub = rospy.Publisher('cmd_vel', Twist)
        rospy.init_node('smartcar_teleop') 
        self.max_tv = rospy.get_param('walk_vel', 0.5)
        self.max_rv = rospy.get_param('yaw_rate', 1.0) 
        self.cmd.linear.x = speed * max_tv; 
        self.cmd.angular.z = turn * max_rv;  

    
    def camera_observe(self,scale=1):
        if self.cap.isOpened():
            ret,frame = self.cap.read()
            frame = cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            print("frame:",frame)
            b,g,r = cv2.split(frame)
            frame =cv2.merge([r,g,b])

            return frame/255.

    def moveCar(self):
        self.cmd.linear.x = 0.0  
        self.cmd.angular.z = 0.0  
        self.pub.publish(cmd)  

    def forward(self):
        self.max_tv = walk_vel_  
        self.speed = 1  
        self.turn = 0 
        self.moveCar()

    def backward(self):
        self.max_tv = walk_vel_  
        self.speed = -1  
        self.turn = 0
        self.moveCar()

    def left(self):
        self.max_rv = yaw_rate_  
        self.speed = 0  
        self.turn = 1 
        self.moveCar()

    def right (self):
        self.max_rv = yaw_rate_  
        self.speed = 0  
        self.turn = -1 
        self.moveCar()

    def stop(self):
        self.max_tv = 0 
        self.speed = 0 
        self.turn = 0
        self.moveCar()

    def stop_robot(self):  
        self.cmd.linear.x = 0.0  
        self.cmd.angular.z = 0.0  
        self.pub.publish(cmd)  

    def image_action(self,response_action):
        if response_action == 0:
            self.forward()
        elif response_action == 1:
            self.backward()
        elif response_action == 2:
            self.left()
        elif response_action == 3:
            self.right()
        else:
            self.stop()
# if __name__ == "__main__":
#     try:
#         car = RosCar()
#         car.forward()
#     except KeyboardInterrupt:
#         pass
    
# def keyboardLoop():  
#     #初始化  
#     rospy.init_node('smartcar_teleop')  
#     rate = rospy.Rate(rospy.get_param('~hz', 1))  
  
#     #速度变量  
#     walk_vel_ = rospy.get_param('walk_vel', 0.5)  
#     run_vel_ = rospy.get_param('run_vel', 1.0)  
#     yaw_rate_ = rospy.get_param('yaw_rate', 1.0)  
#     yaw_rate_run_ = rospy.get_param('yaw_rate_run', 1.5)  
  
#     max_tv = walk_vel_  
#     max_rv = yaw_rate_  
  
#     #显示提示信息  
#     print "Reading from keyboard"  
#     print "Use WASD keys to control the robot"  
#     print "Press Caps to move faster"  
#     print "Press q to quit"  
#     #读取按键循环  
#     while not rospy.is_shutdown():  
#         fd = sys.stdin.fileno()  
#         old_settings = termios.tcgetattr(fd)  
#         #不产生回显效果  
#         old_settings[3] = old_settings[3] & ~termios.ICANON & ~termios.ECHO  
#         try :  
#             tty.setraw( fd )  
#             ch = sys.stdin.read( 1 )  
#         finally :  
#             termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  
  
#         if ch == 'w':  
#             max_tv = walk_vel_  
#             speed = 1  
#             turn = 0  
#         elif ch == 's':  
#             max_tv = walk_vel_  
#             speed = -1  
#             turn = 0  
#         elif ch == 'a':  
#             max_rv = yaw_rate_  
#             speed = 0  
#             turn = 1  
#         elif ch == 'd':  
#             max_rv = yaw_rate_  
#             speed = 0  
#             turn = -1  
#         elif ch == 'W':  
#             max_tv = run_vel_  
#             speed = 1  
#             turn = 0  
#         elif ch == 'S':  
#             max_tv = run_vel_  
#             speed = -1  
#             turn = 0  
#         elif ch == 'A':  
#             max_rv = yaw_rate_run_  
#             speed = 0  
#             turn = 1   
#         elif ch == 'D':  
#             max_rv = yaw_rate_run_  
#             speed = 0  
#             turn = -1  
#         elif ch == 'q':  
#             exit()  
#         else:  
#             max_tv = walk_vel_  
#             max_rv = yaw_rate_  
#             speed = 0  
#             turn = 0  
  
#         #发送消息  
#         cmd.linear.x = speed * max_tv;  
#         cmd.angular.z = turn * max_rv;  
#         pub.publish(cmd)  
#         rate.sleep()  
#         #停止机器人  
#         stop_robot();  
  
# def stop_robot():  
#     cmd.linear.x = 0.0  
#     cmd.angular.z = 0.0  
#     pub.publish(cmd)  
  
# if __name__ == '__main__':  
#     try：
#         roscar()
#     except rospy.ROSInterruptException:
#     pass
    # try:  
    #     keyboardLoop()  
    # except rospy.ROSInterruptException: 
    #     pass
