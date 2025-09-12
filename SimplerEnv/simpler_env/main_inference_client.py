import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
import socket
import pickle
from typing import Sequence
import torch
import cv2 as cv
import matplotlib.pyplot as plt


class ModelSocket:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_to_server()

    def connect_to_server(self):
        connected = False
        while not connected:
            try:
                self.client_socket.connect((self.host, self.port))
                connected = True
                print("Connection established successfully.")
            except (socket.timeout, ConnectionRefusedError) as e:
                print(f"Connection failed: {e}")

    def send_message(self, message):
        message_serialized = pickle.dumps(message)
        self.client_socket.sendall(len(message_serialized).to_bytes(4, byteorder='big'))
        self.client_socket.sendall(message_serialized)

    def receive_message(self):
        length_data = self.client_socket.recv(4)
        if not length_data:
            return None
        data_length = int.from_bytes(length_data, byteorder='big')
        full_data = bytearray()
        while len(full_data) < data_length:
            packet = self.client_socket.recv(data_length - len(full_data))
            if not packet:
                return None
            full_data.extend(packet)
        return pickle.loads(full_data)

    def reset(self, task_description):
        reset_message = {'action':'reset', 'task_description':task_description}
        self.send_message(reset_message)
        state = self.receive_message()
        print('reset process finished YES')

    def step(self, image, task_description, obs):
        step_message = {'action':'step', 'image':image, 'task_description':task_description, 'obs':obs}
        self.send_message(step_message)
        message = self.receive_message()
        return message['raw_action'], message['action']

    def end(self):
        self.client_socket.close()

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        def _resize_image(image: np.ndarray) -> np.ndarray:
            image = cv.resize(image, tuple([224, 224]), interpolation=cv.INTER_AREA)
            return image
        images = [_resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)


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

    model = ModelSocket(host=args.host, port=args.port)
    args.logging_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), "results"+"_"+os.path.splitext(os.path.basename(args.ckpt_path))[0])
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
    model.end()
