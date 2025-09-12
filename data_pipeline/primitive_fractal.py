import numpy as np
from scipy.spatial.transform import Rotation as R

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# to avoid preallocating all GPU memory
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import tensorflow_datasets as tfds

builder = tfds.builder_from_directory(builder_dir='path/to/fractal/dataset')
episode_id = 0
ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
episode = next(iter(ds))



def describe_move(move_vec):
    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "tilt down", 0: None, 1: "tilt up"},  # pitch or roll diff will go here
        {},
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},
        {-1: "open gripper", 0: None, 1: "close gripper"},
    ]

    xyz_move = [names[i][move_vec[i]] for i in range(0, 3)]
    xyz_move = [m for m in xyz_move if m is not None]

    description = "move " + " ".join(xyz_move) if xyz_move else ""

    if move_vec[3] != 0:
        description += (", " if description else "") + names[3][move_vec[3]]
    if move_vec[5] != 0:
        description += (", " if description else "") + names[5][move_vec[5]]
    if move_vec[6] != 0:
        description += (", " if description else "") + names[6][move_vec[6]]

    return description if description else "stop"


def classify_movement(move, threshold=0.02):
    start, end = move[0], move[-1]
    diff = np.zeros(7)  # for move_vec: xyz(3) + pitch(1) + yaw(1) + gripper(1)

    # Δxyz
    xyz_diff = end[:3] - start[:3]
    if np.sum(np.abs(xyz_diff)) > 3 * threshold:
        xyz_diff *= (3 * threshold / np.sum(np.abs(xyz_diff)))
    diff[:3] = xyz_diff

    # Δrotation via quaternion → euler difference
    rot_start = R.from_quat(start[3:7])  # xyzw
    rot_end = R.from_quat(end[3:7])
    delta_rot = rot_end * rot_start.inv()
    euler = delta_rot.as_euler("xyz", degrees=False)  # [roll, pitch, yaw]

    # Use pitch (index 1) and yaw (index 2) for movement detection
    diff[3] = -euler[1]/3  # pitch
    diff[5] = euler[2]/5  # yaw

    # Δgripper
    diff[6] = end[7] - start[7]

    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)
    return describe_move(move_vec), move_vec


def get_move_primitives_episode(states):
    move_actions = dict()
    move_trajs = [states[i: i + 3] for i in range(len(states) - 2)]
    primitives = [classify_movement(move) for move in move_trajs]
    primitives = primitives + [("stop",None)] * 2

    for (desc, _) in primitives:
        if desc in move_actions:
            move_actions[desc].append("move")
        else:
            move_actions[desc] = ["move"]

    return primitives


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from tqdm import trange
from tqdm import tqdm
results = {}

batch_size = 500
total_episodes = 87212

for start in trange(0, total_episodes, batch_size):
    end = min(start + batch_size, total_episodes)
    
    ds = builder.as_dataset(split=f"train[{start}:{end}]", batch_size=None)

    for episode in ds:
        try:
            ep_id = episode['episode_metadata']['episode_id'].numpy().decode()
            states = [
                tf.concat(
                    (step["observation"]["base_pose_tool_reached"], step["observation"]["gripper_closed"]),
                    axis=-1
                ).numpy()
                for step in episode["steps"]
            ]
            primitives = get_move_primitives_episode(states)
            descriptions = [p[0] for p in primitives]
            if ep_id not in results:
                results[ep_id] = {}
            if "features" not in results[ep_id]:
                results[ep_id]["features"] = {}

            results[ep_id]["features"]["move_primitive"] = descriptions

        except Exception as e:
            print(f"[Error] episode {start}–{end} entry: {e}")

with open('fractal_primitive.json','w') as f:
    json.dump(f, results)
