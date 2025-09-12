"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from collections import deque
from vla.instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist import load_vla
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any, Sequence
import numpy as np
import tensorflow as tf
import torch
from PIL import Image

class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)  
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)  
        norm_ref = np.linalg.norm(ref)  
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
  
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")

class InstructVLAServer:
    def __init__(
        self,
        cfg, 
    ) -> None:
        self.cfg = cfg
        self.action_ensemble = cfg.action_ensemble
        self.adaptive_ensemble_alpha = cfg.adaptive_ensemble_alpha
        self.action_ensemble_horizon = cfg.action_ensemble_horizon
        self.horizon = cfg.horizon

        self.task_description = None

        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        
        self.vla = load_vla(
            cfg.pretrained_checkpoint,
            load_for_training=False, 
            action_model_type=cfg.action_model_type,
            future_action_window_size=cfg.future_action_window_size,
            past_action_window_size=cfg.horizon,
            action_dim=cfg.action_dim,
        )
        if self.cfg.use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()
        self.global_step = 0
        self.last_action_chunk = None



    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        if self.action_ensemble:
            self.action_ensembler.reset()

        self.global_step = 0
        self.last_action_chunk = None

    def crop_and_resize(self, image, crop_scale, batch_size):
        """
        Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
        to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
        distribution shift at test time.

        Args:
            image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
                values between [0,1].
            crop_scale: The area of the center crop with respect to the original image.
            batch_size: Batch size.
        """
        # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
        assert image.shape.ndims == 3 or image.shape.ndims == 4
        expanded_dims = False
        if image.shape.ndims == 3:
            image = tf.expand_dims(image, axis=0)
            expanded_dims = True

        # Get height and width of crop
        new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
        new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

        # Get bounding box representing crop
        height_offsets = (1 - new_heights) / 2
        width_offsets = (1 - new_widths) / 2
        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )

        # Crop and then resize back up
        image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

        # Convert back to 3D Tensor (H, W, C)
        if expanded_dims:
            image = image[0]

        return image


    def get_cronusvla_action(self, vla, cfg, base_vla_name, obs, task_label, unnorm_key, center_crop=True):
        """Generates an action with the VLA policy."""
        all_images = [obs["full_image"]]
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        processed_images = []
        for image in all_images:
            pil_image = Image.fromarray(image).convert("RGB")

            if center_crop:
                pil_image = center_crop_image(pil_image)
            
            processed_images.append(pil_image)
            

        if task_label is not None:
            if task_label.lower() != self.task_description:
                self.reset(task_label.lower())

        if self.cfg.use_length == -1 or self.global_step % self.cfg.use_length == 0:
            action, normalized_actions, cognition_features_current = vla.predict_action(image=processed_images, 
                                                                            instruction=self.task_description,
                                                                            unnorm_key=unnorm_key,
                                                                            do_sample=False,
                                                                            )
            self.last_action_chunk = action
        
        if self.cfg.use_length > 0:
            action = self.last_action_chunk[self.global_step % self.cfg.use_length]
        elif self.cfg.use_length == -1: # do ensemble
            action = self.action_ensembler.ensemble_action(action)

        self.global_step+=1

        return action


