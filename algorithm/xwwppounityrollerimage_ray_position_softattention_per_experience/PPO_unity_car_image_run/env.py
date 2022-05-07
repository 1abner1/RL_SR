'''
可以存在缺帧，每次接收到动作后才会更改动作
    1 reset 得到 obsrv
    2 obsrv → 算法
    3 算法 → 动作
    4 动作 → action()
    5 action() → obsr
    奖励 
    unity交互
    ip池(多艇，真假艇混合)
    f字符串
    任务类型3  单艇巡线避障 终点已知 障碍 物类型中某一种定义为目标
    任务类型4  动态目标追踪
'''
import time
import json
import socket
import random

class Env:
    def __init__(self):
        self.BUFSIZE = 1024 
        #self.sever_ip = '192.168.4.2' #总控ip
        self.sever_ip = '127.0.0.1'
        self.rec_port = 7001 #接受端口 7001
        self.send_port = 6001 #发送端口 6001
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # udp协议
        self.server.bind((self.sever_ip,self.rec_port))
        self.client = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.recvmsg = ''
        self.sendmsg = ''
        self.count = 0
        self.obsrv = [0,0,0,0,0,0,0,0,0,0] #给算法的观测值 0-2无人艇位置 3-5目标位置 6-7xz方向速度 8-9xz方向
        self.msg21 = []
        self.msg22 = [22,1,3,2,0,0,0,0] #动作
        self.msg23 = []
    
    def sendMsg(self,msg):   #msg是列表
        self.sendmsg = '[' + ','.join([str(t) for t in msg]) +']'
        self.client.sendto(self.sendmsg.encode('utf-8'),(self.sever_ip,self.send_port))
        time.sleep(0.1)      #程序停止运行0.1秒，后续进行调整，使得动作频率为10HZ

    def recMsg(self):
        self.recvmsg = self.sever.recv(self.BUFSIZE)
        self.recvdata = []
        self.recvdata = self.recvmsg.split(',',-1)
        return self.recvdata

    def heartBeat(self):
        self.sendmsg = f'[17,{self.count}]'
        self.count = (self.count + 1) % 10000
        self.sendMsg(self.sendmsg)

    def reset(self):
        res = []
        self.heartBeat() #发送心跳包，表示开始
        t = 0
        while 1:
            res = self.recMsg()
            if res[0] == 21:
                if len(self.msg21) == 0:
                     t = t + 1
                self.msg21 = res
            elif res[0] == 23:
                if len(self.msg23) == 0:
                    t = t + 1
                self.msg23 = res
            else:
                pass
            if t == 2:
                break
        return self.getObsrv()

    def step(self,actions):
        self.sendAction(actions)
        return  self.reset(),0,0,"NA"

    def getObsrv(self): #假设无人艇经纬范围： 5-10;5-10
        self.obsrv[0] = (self.msg21[2] - 5) / 5 * 80 - 40
        self.obsrv[1] = 0.2
        self.obsrv[2] = (self.msg21[3] - 5) / 5 * 80 - 40
        self.obsrv[3] = (self.msg23[3] - 5) / 5 * 80 - 40
        self.obsrv[4] = 0.2
        self.obsrv[5] = (self.msg23[4] - 5) / 5 * 80 - 40
        self.obsrv[6] = 0
        self.obsrv[7] = 0
        self.obsrv[8] = 0
        self.obsrv[9] = 0
        return self.obsrv

    def sendAction(self,action):
        self.msg22[5] = action[0] * 100
        self.msg22[6] = action[1] * 30
        self.sendMsg(self.msg22)
    def init(self):
        return len(self.obsrv),0,2


def main():
    ks = Env()
    while 1:
        ks.reset()
        ks.sendMsg(ks.obsrv)
        ac = [0,0]
        ac[0] = random.randint(0,1)
        ac[1] = random.randint(-1,1)
        ks.sendAction(ac)


if __name__ == "__main__":
    # execute only if run as a script
    main()


    
