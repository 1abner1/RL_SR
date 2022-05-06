import server
import grpc
from concurrent import futures
import ros_message_pb2_grpc
import ros_message_pb2

# sr1 = server.RosMessageServicer()
# sr2 = sr1.GetAction
# print("sr2", sr2)

# request.observations[0]
# print("t11111111111111111111",t)
# start_communcation1 = sr1.start_communcation()
# print("开始通信1111111111111111111")
# def init(self):
#     obs_shapes = 1
#     discrete_action_size = 2
#     continuous_action_size= 3
#     return obs_shapes, discrete_action_size, continuous_action_size
# def reset(self):
#     pass
# def step(self, d_action, c_action):
#     obs_ = 1
#     reward = 2
#     return obs_, reward
# def close():
#     pass

class ros_wrapper():
    def __init__(self):
        self.servicer =server.RosMessageServicer()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                             options=[('grpc.max_send_message_length', 8388608),
                                      ('grpc.max_receive_message_length', 8388608)])
        ros_message_pb2_grpc.add_RosMessageServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f'[::]:8888')
        t = self.server.start()
    def get_ros_obsvation(self):
        self.server.wait_for_termination(0.1)
        return self.servicer.vis

tt = ros_wrapper()
while(1):
    tt1 = tt.get_ros_obsvation()
    print("tt1:",tt1)