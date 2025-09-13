"""
instructvla_policy_state.py

"""
from collections import deque
from typing import Optional, Sequence
import os
from PIL import Image
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2_state import load_vla
from .adaptive_ensemble import AdaptiveEnsembler
from .geometry import euler2axangle, mat2euler, quat2mat
from copy import deepcopy
import pickle

class InstructVLAInference:
    def __init__(
        self,
        saved_model_path: str = 'TBD',
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        future_action_window_size: int = 15,
        action_dim: int = 7,
        action_model_type: str = "DiT-B",
        action_scale: float = 1.0,
        use_bf16: bool = True,
        action_ensemble = True,
        adaptive_ensemble_alpha = 0.1,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_dataset" if unnorm_key is None else unnorm_key
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 7 for widowx_bridge to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 2 for google_robot to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.vla = load_vla(
          saved_model_path,
          load_for_training=False, 
          future_action_window_size=future_action_window_size,
          past_action_window_size=horizon,
          action_dim=action_dim,
        )

        if use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.cognition_features_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.num_cognition_features_history = 0

    def _add_cognition_features_to_history(self, cognition_feature) -> None:
        self.cognition_features_history.append(cognition_feature)
        self.num_cognition_features_history = min(self.num_cognition_features_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.cognition_features_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.num_cognition_features_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def preprocess_proprio(self, obs: dict) -> np.array:
        # convert ee rotation to the frame of top-down
        if self.policy_setup == "widowx_bridge":
            self.default_rot = np.array(
                [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
            )
            proprio = obs["agent"]["eef_pos"]
            rm_bridge = quat2mat(proprio[3:7])
            rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
            gripper_openness = proprio[7]
            raw_proprio = np.concatenate(
                [
                    proprio[:3],
                    rpy_bridge_converted,
                    np.zeros(1),
                    [gripper_openness],
                ]
            )
        elif self.policy_setup == "google_robot":
            """convert wxyz quat from simpler to xyzw used in fractal"""
            quat_xyzw = np.roll(obs["agent"]["eef_pos"][3:7], -1)
            gripper_width = obs["agent"]["eef_pos"][
                7
            ]  # from simpler, 0 for close, 1 for open
            gripper_closedness = (
                1 - gripper_width
            )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
            raw_proprio = np.concatenate(
                (
                    obs["agent"]["eef_pos"][:3],
                    quat_xyzw,
                    [gripper_closedness],
                )
            )
        return raw_proprio

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, obs = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image: Image.Image = Image.fromarray(image)
        raw_actions, normalized_actions, cognition_features_current = self.vla.predict_action(image=image, 
                                                                        instruction=self.task_description,
                                                                        unnorm_key=self.unnorm_key,
                                                                        do_sample=False, 
                                                                        proprio = self.preprocess_proprio(obs),
                                                                        )
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
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