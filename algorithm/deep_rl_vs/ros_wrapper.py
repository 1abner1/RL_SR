import server
import grpc
from concurrent import futures
import ros_message_pb2_grpc
import ros_message_pb2
from numproto import ndarray_to_proto, proto_to_ndarray

class ros_wrapper():
    def __init__(self):
        self.servicer =server.RosMessageServicer()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                             options=[('grpc.max_send_message_length', 8388608),
                                      ('grpc.max_receive_message_length', 8388608)])
        ros_message_pb2_grpc.add_RosMessageServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f'[::]:8888')
        t = self.server.start()
        # self.action1 =[]
        print("服务准备开始")
        self.server.wait_for_termination()

    # def get_ros_obsvation(self):
    #     # self.server.wait_for_termination(0.1)
    #     # print("开始获得ros 观察值")
    #     return self.servicer.vis
    # def recv_action(self,action):
    #     # self.servicer.action = action
    #     ros_message_pb2.ActionResponse(action=ndarray_to_proto(action))
    #     return ros_message_pb2.ActionResponse(action=ndarray_to_proto(action))
    # def step(self,ros_action):
    #     nex_obs = None
    #     if ros_action == 4:
    #         reward = [-1]
    #     elif ros_action == 0:
    #         reward = [1]
    #     done = False
    #     _ = None
    #     return nex_obs,reward,done,_

#ros_wrapper api代码，通信断了之后机器人会一直重复最后一张照片
# ros_env = ros_wrapper()
# while(1):
#     ros_obs = ros_env.get_ros_obsvation()
#     ros_env.servicer.action = [0]
#     print("ros_obs",ros_obs)
#     # 显示2s 图像
#     from matplotlib import pyplot as plt
#     # plt.switch_backend("agg")
#     plt.imshow(ros_obs)
#     plt.show(block=False)
#     plt.pause(0.05)
#     plt.close()
#最新输出运行的代码
# ros_env = ros_wrapper()