import rospy
from sensor_msgs.msg import LaserScan

class LaserScanData:
    def __init__(self, callback):
        self.callback = callback

    def callback_(self, data):
        self.callback(data)

    def start_listen(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/scan", LaserScan, self.callback_)
        rospy.spin()


if __name__ == '__main__':
    def laser_data(data):
        print(data)
    a = LaserScanData(laser_data)
    a.start_listen()
