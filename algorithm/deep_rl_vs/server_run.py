import sys
sys.path.append(r"D:\Pytorch_RL _sort\algorithm\deep_rl\agents")
from concurrent import futures
import grpc
import numpy as np
import ros_message_pb2
import ros_message_pb2_grpc
from numproto import ndarray_to_proto, proto_to_ndarray
import matplotlib.pyplot as plt
import cv2
import run_unity_class

class RosMessageServicer(ros_message_pb2_grpc.RosMessageServicer):
    def __init__(self):
        self.vis = np.zeros([84,84,3])
        self.action = []

    def Init(self, request: ros_message_pb2.InitRequest, context):
        # print('observation shapes')
        self.vis = np.zeros(1)
        for s in request.observation_shapes:
            print('observation shapes:', s.dim)
        try:
            # fig, ax = plt.subplots()
            # self.im = ax.imshow(np.zeros(request.observation_shapes[0].dim))
            pass
        except Exception as e:
            print(e)

        # obs_image = proto_to_ndarray(request.observations[0])
        # print("初始化时获取图像数据",obs_image)
        return ros_message_pb2.Empty()
        # return obs_image

    def GetAction(self, request: ros_message_pb2.ObservationsRequest, context):
        self.vis = proto_to_ndarray(request.observations[0])

        print("获得观察图像", self.vis)
        self.action = [0]
        # 显示2s 图像
        from matplotlib import pyplot as plt
        plt.imshow(self.vis[:,:,[2,1,0]])
        plt.show(block=False)
        plt.pause(0.05)
        plt.close()
        #运行算法端
        vis_obs = self.vis
        real_to_car = run_unity_class.run_unity_rl.main(vis_obs)
        self.acton = real_to_car
        run_unity_class.run_unity_rl.main()




        # ros_obs = ros_env.get_ros_obsvation()
        # ros_env.servicer.action = [0]
        # print("ros_obs", ros_obs)
        # # 显示2s 图像
        # from matplotlib import pyplot as plt
        # # plt.switch_backend("agg")
        # plt.imshow(ros_obs)
        # plt.show(block=False)
        # plt.pause(0.05)
        # plt.close()


        return ros_message_pb2.ActionResponse(action=ndarray_to_proto(self.action))

<<<<<<< HEAD
    def start_communcation():
=======
    def start_communcation(self):
>>>>>>> master
        servicer = RosMessageServicer()
        print("创建一个服务1111111111111111111")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                             options=[('grpc.max_send_message_length', 8388608),
                                      ('grpc.max_receive_message_length', 8388608)])
        ros_message_pb2_grpc.add_RosMessageServicer_to_server(servicer, server)
        # print("开始初始化，已经获得动作")
        server.add_insecure_port(f'[::]:8888')
        t = server.start()
        print("服务开始", t)
<<<<<<< HEAD
        server.wait_for_termination()
        print("服务终止1111111111111111111111", t)
        # server.wait_for_termination(0.1)
        # print("服务开始11111111111111111")
        return print("通信结束了")
c = RosMessageServicer.start_communcation()

# 在外面运行
=======
        server.wait_for_termination(0.1)
        print("服务终止1111111111111111111111", t)
        server.wait_for_termination(0.1)
        print("服务开始11111111111111111")
        # return print("通信结束了")
# c = start_communcation()
>>>>>>> master
# servicer = RosMessageServicer()
# print("创建一个服务1111111111111111111")
# server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=[('grpc.max_send_message_length',8388608),('grpc.max_receive_message_length',8388608)])
#
# ros_message_pb2_grpc.add_RosMessageServicer_to_server(servicer, server)
# server.add_insecure_port(f'[::]:8888')
# # a =self.get_obs(request: ros_message_pb2.ObservationsRequest, context)
# # print('aaaaaaaaaaaaa',a)
#  # print('aaaaaaaaaaaaa',a)
# t = server.start()
# print("服务开始")
# server.wait_for_termination()
