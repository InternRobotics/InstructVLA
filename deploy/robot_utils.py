"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch

from deploy.openvla_utils import (
    get_vla,
    get_vla_action,
)
from deploy.instructvla_utils import (
    InstructVLAServer
)

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})



def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""
    if cfg.model_family == "openvla":
        cronus_server = None
        model = get_vla(cfg)
    elif cfg.model_family == "instruct_vla":
        cronus_server = InstructVLAServer(cfg)
        model = cronus_server.vla
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model, cronus_server


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla" or "instruct_vla" in cfg.model_family:
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, server, processor=None):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
    elif "instruct_vla" in cfg.model_family:
        action = server.get_cronusvla_action(
            model, cfg, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        # assert action.shape == (ACTION_DIM,), 'action shape is wrong'
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
