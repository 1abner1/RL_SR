import time
import grpc
import numpy as np
import ros_message_pb2
import ros_message_pb2_grpc
from ros_car import Roscar
from numproto import *

VISUAL_SCALE = 1
DIRECTION_MIN_INTERVAL = 0.2
if __name__ == "__main__":
    car = Roscar()
    channel = grpc.insecure_channel('192.168.31.184:8888',
                                    [('grpc.max_reconnect_backoff_ms', 5000),('grpc.max_send_message_length',8388608),('grpc.max_receive_message_length',8388608)])
    stub = ros_message_pb2_grpc.RosMessageStub(channel)
    try:
        vis = car.camera_observe(VISUAL_SCALE)
        response = stub.Init(ros_message_pb2.InitRequest(observation_shapes=[ros_message_pb2.Shape(dim=list(vis.shape))],
                        action_shape=ros_message_pb2.Shape(dim=[5, ])))
    except Exception as e:
        print(e)
    while 1:
        try:
            vis = car.camera_observe(VISUAL_SCALE)
            response = stub.GetAction(ros_message_pb2.ObservationsRequest(observations=[ndarray_to_proto(vis)]))
            action = proto_to_ndarray(response.action)
            print(action)
            car.image_action(action)
        except Exception as e:
            car.image_action(4)
            print(e)
            break