import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
import socket
import pickle

class ServerSocket:
    def __init__(self, model=None, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.model = model
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print("Server listening on", self.host, self.port)

        while True:
            conn, addr = self.server_socket.accept()
            self.handle_client(conn, addr)

    def send_message(self, conn, message):
        message_serialized = pickle.dumps(message)
        conn.sendall(len(message_serialized).to_bytes(4, byteorder='big'))
        conn.sendall(message_serialized)

    def receive_message(self, conn):
        length_data = conn.recv(4)
        if not length_data:
            return None
        data_length = int.from_bytes(length_data, byteorder='big')
        full_data = bytearray()
        while len(full_data) < data_length:
            packet = conn.recv(data_length - len(full_data))
            if not packet:
                return None
            full_data.extend(packet)
        return pickle.loads(full_data)

    def handle_client(self, conn, addr):
        print(f"Connected by {addr}")
        with conn:
            while True:
                data = self.receive_message(conn)
                if not data:
                    print(f"Connection closed by {addr}")
                    break
                try:
                    if data['action'] == 'reset':
                        self.model.reset(data['task_description'])
                        state = 'YES'
                        self.send_message(conn, state)
                    elif data['action'] == 'step':
                        raw_action, action = self.model.step(data['image'], data['task_description'])
                        message = {'raw_action':raw_action, 'action':action}
                        self.send_message(conn, message)
                except Exception as e:
                    print(f"Error handling data from {addr}: {e}")
                    break

if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    # policy model creation; update this if you are using a new policy model
    args.policy_model = "cogact"
    assert args.policy_model == "cogact"
    if "meta" in args.ckpt_path:
        from simpler_env.policies.sim_cogact import MetaCogACTInference
        assert args.ckpt_path is not None
        model = MetaCogACTInference(
            saved_model_path=args.ckpt_path,  # e.g., CogACT/CogACT-Base
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_model_type='DiT-B',
            cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
        )
    else:
        raise NotImplementedError('The code is only varified using meta token method.')
        from simpler_env.policies.sim_cogact import CogACTInference
        assert args.ckpt_path is not None
        model = CogACTInference(
            saved_model_path=args.ckpt_path,  # e.g., CogACT/CogACT-Base
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_model_type='DiT-B',
            cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
        )
    server = ServerSocket(model=model, host=args.host, port=args.port)
    server.start()
