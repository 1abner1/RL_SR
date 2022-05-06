import grpc
import numpy as np

import ros_message_pb2
import ros_message_pb2_grpc
from numproto import ndarray_to_proto, proto_to_ndarray


channel = grpc.insecure_channel('192.168.1.140:8888')
stub = ros_message_pb2_grpc.RosMessageStub(channel)
response = stub.Init(ros_message_pb2.InitRequest(
    observation_shapes=[ros_message_pb2.Shape(dim=[1, 2]),
                        ros_message_pb2.Shape(dim=[1, 2, 3])],
    action_shape=ros_message_pb2.Shape(dim=[1, 10])
))
# print("client is ok")
while True:
    # print("client is ok")
    input()
    print("client is ok")
    response = stub.GetAction(ros_message_pb2.ObservationsRequest(
        observations=[
            ndarray_to_proto(np.random.randn(10, 5)),
            ndarray_to_proto(np.random.randn(10, 2))
        ]
    ))

    print(proto_to_ndarray(response.action))
    print("client is ok")
