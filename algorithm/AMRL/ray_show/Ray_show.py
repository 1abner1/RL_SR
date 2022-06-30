import matplotlib
import matplotlib.pyplot as plt
import math


class Ray_show():
    MAX_ANGLE = math.pi * 2/3
    per_side_length = 0
    per_angle = 0
    left_sin = []
    left_cos = []
    right_sin = []
    right_cos = []

    def __init__(self, ray_num):
        matplotlib.rcParams['toolbar'] = 'None'
        plt.figure('Draw')
        plt.ion()
        ax = plt.gca()
        ax.set_frame_on(False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        self.per_side_length = int(ray_num/2)
        self.per_angle = self.MAX_ANGLE/self.per_side_length
        for i in range(ray_num):
            if i == 0:
                pass
            elif i % 2 == 0:
                num = i/2
                self.left_sin.append(math.sin(2*math.pi - num*self.per_angle))
                self.left_cos.append(math.cos(2*math.pi - num*self.per_angle))
            else:
                num = (i+1)/2
                self.right_sin.append(math.sin(num*self.per_angle))
                self.right_cos.append(math.cos(num*self.per_angle))

    def show(self, ray):
        plt.cla()
        plt.plot(0, 1, 'bo')
        plt.plot(0, -1, 'bo')
        plt.plot(1, 0, 'bo')
        plt.plot(-1, 0, 'bo')
        for i in range(len(ray)):
            if(ray[i] == 1):
                continue
            if i == 0:
                x = 0
                y = ray[i]
            elif i % 2 == 0:
                num = int(i/2)
                x = self.left_sin[num-1]*ray[i]
                y = self.left_cos[num-1]*ray[i]
            else:
                num = int((i+1)/2)
                x = self.right_sin[num-1]*ray[i]
                y = self.right_cos[num-1]*ray[i]
            plt.plot(x, y, 'r.')
        plt.plot(0, 0, 'b+')
        plt.pause(0.0001)
        # plt.savefig('test.png', bbox_inches='tight', pad_inches=0)
#开始运行
#ray_figure = Ray_show(len(ray))
#ray_figure.show(ray)
