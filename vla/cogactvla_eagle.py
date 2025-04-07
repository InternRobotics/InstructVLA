"""
cogactvla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any, Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

from action_model.action_model import ActionModel
import torch.nn.functional as F

from vla.eagle_utils import EagleProcessor
from types import SimpleNamespace

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


def model_forward(self):
    def forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = None # self.lm_head(hidden_states[:, slice_indices, :]) # remove lm head for faster vla training

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    return forward

class ActionModel(nn.Module):
    def __init__(self, 
                 token_size,
                 past_action_window_size,
                 future_action_window_size,
                 action_dim
                 ):
        super().__init__()
        self.action_model = nn.Sequential(  nn.Linear(token_size * (past_action_window_size + 1), token_size*4, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(token_size*4, token_size*8, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(token_size*8, token_size*8, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(token_size*8, token_size*4, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(token_size*4, token_size*2, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(token_size*2, token_size*2, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(token_size*2, token_size//2, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(token_size//2, (future_action_window_size + 1) * action_dim, bias=True),
                                        ) # 120M
    def forward(self,
                cognition_features):
        return self.action_model(cognition_features)


class CogACT(nn.Module):
    def __init__(
        self,
        vlm: AutoModel,
        processor: AutoProcessor = None,
        tokenizer: AutoTokenizer = None,
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 1,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        config_json = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.action_model = ActionModel(token_size,past_action_window_size,future_action_window_size,action_dim)
        
        self.vlm = vlm
        self.processor = processor
        self.tokenizer = tokenizer
        self.config_json = config_json
        self.token_size = token_size
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ['action_model', 'ema_diffusion']
        else:
            self.all_module_keys = ['action_model']
        for module_keys in ["vision_model", "language_model", "mlp1"]:
            self.all_module_keys.append("vlm." + module_keys)

        # Diffusion head is always trainable
        self._trainable_module_keys = ['action_model']

        keys = []
        for module_keys in ["vision_model", "language_model", "mlp1"]:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        self.trainable_module_keys = keys

        self.norm_stats = norm_stats
        self.vlm.model = self.vlm.language_model
        self.vlm.model.forward = model_forward(self.vlm.model)
        self.vlm.model.transformer_layer_cls = Qwen2DecoderLayer
        self.vlm.neftune_alpha = None


    @property
    def llm_backbone(self):
        return self.vlm.model
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        per_device_batch_size: int = 16,
        action_masks = None,
        **kwargs,
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        assert per_device_batch_size == actions.shape[0]
        assert input_ids.shape[0] == (per_device_batch_size * (self.past_action_window_size+1))

        indices_for_past = torch.cat([torch.arange(i*(self.past_action_window_size+1), i*(self.past_action_window_size+1)+self.past_action_window_size) for i in range(per_device_batch_size)])
        indices_for_now = torch.arange(self.past_action_window_size, per_device_batch_size*(self.past_action_window_size+1), self.past_action_window_size+1)

        with torch.no_grad():
            output_for_past: CausalLMOutputWithPast = self.vlm(
                input_ids=input_ids[indices_for_past] if input_ids is not None else input_ids,
                attention_mask=attention_mask[indices_for_past] if attention_mask is not None else attention_mask,
                pixel_values=pixel_values[indices_for_past],
                # labels=labels[indices_for_past] if labels is not None else labels,
                # inputs_embeds=inputs_embeds[indices_for_past] if inputs_embeds is not None else inputs_embeds,
                image_flags=torch.ones((indices_for_past.shape[0],1)).to( device=input_ids.device),
                past_key_values=past_key_values[indices_for_past] if past_key_values is not None else past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_for_past = output_for_past.hidden_states[-1]
        last_hidden_for_past = last_hidden_for_past.clone().detach().requires_grad_()
        assert last_hidden_for_past.shape[0] == (per_device_batch_size * (self.past_action_window_size))
        last_hidden_for_past = last_hidden_for_past.reshape(per_device_batch_size, self.past_action_window_size, -1, self.token_size)

        output_for_now: CausalLMOutputWithPast = self.vlm(
            input_ids=input_ids[indices_for_now] if input_ids is not None else input_ids,
            attention_mask=attention_mask[indices_for_now] if attention_mask is not None else attention_mask,
            pixel_values=pixel_values[indices_for_now],
            # labels=labels[indices_for_now] if labels is not None else labels,
            # inputs_embeds=inputs_embeds[indices_for_now] if inputs_embeds is not None else inputs_embeds,
            image_flags=torch.ones((indices_for_now.shape[0],1)).to( device=input_ids.device),
            past_key_values=past_key_values[indices_for_now] if past_key_values is not None else past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_for_now = output_for_now.hidden_states[-1]
        assert last_hidden_for_now.shape[0] == (per_device_batch_size * 1)
        last_hidden_for_now = last_hidden_for_now.reshape(per_device_batch_size, 1, -1, self.token_size)

        last_hidden = torch.cat([last_hidden_for_past, last_hidden_for_now], dim=1).reshape(per_device_batch_size * (self.past_action_window_size + 1), -1, self.token_size)
        assert last_hidden.shape[0] == per_device_batch_size * (self.past_action_window_size+1)

        # extract the cognition feature
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B*C, 1, D]
        cognition_features = cognition_features.view(per_device_batch_size,self.past_action_window_size+1,1,-1).squeeze(2) # [B, C, D]
        BS, step, dim = cognition_features.shape
        # actions_history = actions[:,0:self.past_action_window_size,:]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]

        _, _, action_dim = actions_future.shape

        pred = self.action_model(cognition_features.reshape(BS, -1)).reshape(BS, self.future_action_window_size + 1, action_dim)
        loss = F.mse_loss(pred, actions_future)
        return loss, None

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT

        vit_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={SiglipEncoderLayer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer})

        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, ActionModel},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vit_wrap_policy,
                transformer_block_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 5,
        use_ema: bool = False,
        norm_stats = None,
        **kwargs,
    ) -> CogACT:

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = AutoModel.from_pretrained('/mnt/petrelfs/yangshuai1/yangshuai1/share_mllm/Eagle2-2B',
                                        attn_implementation="flash_attention_2",
                                        trust_remote_code=True)

        processor = EagleProcessor(
            '/mnt/petrelfs/yangshuai1/yangshuai1/share_mllm/Eagle2-2B',
            max_input_tiles=1,
            model_spec=SimpleNamespace(
                num_image_token = 256,
                template = "qwen2-chat"
            ),
        )

        tokenizer = AutoTokenizer.from_pretrained('/mnt/petrelfs/yangshuai1/yangshuai1/share_mllm/Eagle2-2B', 
                                                  use_fast=True,
                                                  trust_remote_code=True)

        vlm.img_context_token_id = processor.get_img_context_token()
        assert vlm.template == processor.model_spec.template
        assert vlm.num_image_token == processor.model_spec.num_image_token

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize CogACT
        cogact = CogACT(vlm,
                        processor = processor,
                        tokenizer = tokenizer,
                        token_size = vlm.config.llm_config.hidden_size,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        past_action_window_size = past_action_window_size,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        )
        
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]

        assert (
            "mlp1" in model_state_dict and "language_model" in model_state_dict and "vision_model" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector`, `language_model` AND `vision_model`"

        vlm.mlp1.load_state_dict(model_state_dict["mlp1"])
        vlm.language_model.load_state_dict(model_state_dict["language_model"])
        if "vision_model" in model_state_dict.keys():
            vlm.vision_model.load_state_dict(model_state_dict["vision_model"])

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            cogact.action_model.load_state_dict(model_state_dict["action_model"])
            if "ema_diffusion" in model_state_dict and use_ema:
                cogact.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
            elif use_ema:
                cogact.ema_diffusion.load_state_dict(model_state_dict["action_model"])
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")
        return cogact       

    @torch.inference_mode()
    def predict_action(
        self, 
        image: Image, 
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        cognition_features_history = None,
        num_cognition_features_history = 0,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        # Build VLA Prompt
        prompt = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": f"What action should the robot take to {instruction.lower()}?",
                "image": [{'np_array': np.asarray(image)}],
            },
            {
                "role": "assistant", 
                "content": ""
            }
        ]
        # Prepare Inputs

        inputs = self.processor.prepare_input({"prompt": prompt})
        input_ids = inputs['input_ids'].to(self.vlm.device)
        pixel_values = inputs['pixel_values']

        # Preprocess Image
        pixel_values = pixel_values.to(self.vlm.device)
        
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = torch.bfloat16

        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
            attention_mask = input_ids.ne(-10)
            output: CausalLMOutputWithPast = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_flags=torch.ones((input_ids.shape[0],1)).to( device=input_ids.device),
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None,
            )

        # Extract cognition feature
        cognition_features = output.hidden_states[-1][0,-1:]
        assert (cognition_features.shape[0], cognition_features.shape[1]) == (1,self.token_size), "Batch size must be 1 for action prediction"
        using_cfg = cfg_scale > 1.0

        model_dtype = next(self.action_model.parameters()).dtype
        B = cognition_features.shape[0]

        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]
        cognition_features_copy = cognition_features.clone()

        repeat_num = min(max(0, self.past_action_window_size-num_cognition_features_history), self.past_action_window_size)
        cognition_features_history = list(cognition_features_history)+[cognition_features]
        cognition_features = torch.cat([cognition_features_history[0]]*repeat_num+cognition_features_history, dim=1)
        # Sample random noise
        BS, step, dim = cognition_features.shape
        samples = self.action_model(cognition_features.reshape(BS, -1)).reshape(BS, self.future_action_window_size + 1, 7)
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions, cognition_features_copy

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSDataset
from transformers import AutoTokenizer, AutoProcessor
from huggingface_hub import HfFileSystem, hf_hub_download
import os
import json

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    processor: AutoProcessor

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        
        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        if rlds_batch["observation"]["image_primary"].shape[0] > 1:
            img = rlds_batch["observation"]["image_primary"]
        else:
            raise ValueError(f"Multiimage required")

        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        pixel_values = []
        input_ids = []

        for i in img:
            prompt = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": f"What action should the robot take to {lang}?",
                    "image": [{'np_array': i}],
                },
                {
                    "role": "assistant", 
                    "content": ""
                }
            ]

            inputs = self.processor.prepare_input({"prompt": prompt})

            pixel_values.append(inputs['pixel_values'])
            input_ids.append(inputs['input_ids'])

        pixel_values = torch.stack(pixel_values)
        input_ids = torch.cat(input_ids, dim=0)

        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=input_ids, 
                    dataset_name=dataset_name, actions=action, action_masks=action_mask, 
                    episode_idx=rlds_batch["idx"], frame_idx=rlds_batch['frame_idx'],
                    )

@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        batch_pixel_values   = [inst["pixel_values"]   for inst in instances]  # List[Tensor], each [M_i, C, H, W]
        batch_input_ids      = [inst["input_ids"]      for inst in instances]  # List[Tensor], each [M_i, seq_len]
        batch_labels         = [inst["labels"]         for inst in instances]  


        assert self.padding_side == "right", f"Invalid Tokenizer `padding_side={self.padding_side}`, must be 'right'"
        all_input_ids = []
        all_labels    = []
        for i in range(len(batch_input_ids)):
            for row_ids in batch_input_ids[i]:   # shape (seq_len_i,)
                all_input_ids.append(row_ids)
            for row_lbl in batch_labels[i]:      # shape (seq_len_i,)
                all_labels.append(row_lbl)

        input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels    = pad_sequence(all_labels,    batch_first=True, padding_value=IGNORE_INDEX)

        # truncate if needed: model_max_length
        if input_ids.shape[1] > self.model_max_length:
            input_ids = input_ids[:, : self.model_max_length]
            labels    = labels[:, : self.model_max_length]

        attention_mask = input_ids.ne(self.pad_token_id)
        pixel_values = torch.cat(batch_pixel_values, dim=0).to(self.pixel_values_dtype).squeeze(1)

        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # Adding continuous actions and batch processing.
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions)
        action_masks = [instance["action_masks"] for instance in instances]
        action_masks = torch.stack(action_masks)

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_masks=action_masks,
            episode_idx=[instance['episode_idx'] for instance in instances] if 'episode_idx' in instances[0] else None,
            frame_idx=[instance['frame_idx'] for instance in instances] if 'frame_idx' in instances[0] else None,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    tokenizer: PreTrainedTokenizerBase,
    processor: AutoProcessor,
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,         # Concatenated `past_action_window_size-1' actions and the current action for the input
    load_all_data_for_training: bool = True,  # Load all data for training, or only a subset
    base_action_tokenizer: PreTrainedTokenizerBase = None
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    if base_action_tokenizer is None:
        action_tokenizer = None
    else:
        action_tokenizer = ActionTokenizer(base_action_tokenizer)
    # action_tokenizer = ActionTokenizer(tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, processor
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        future_action_window_size=future_action_window_size,
        past_action_window_size=past_action_window_size,
        image_aug=image_aug,
        load_all_data_for_training=load_all_data_for_training,
    )

    return dataset, action_tokenizer, collator


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    llm_backbone_id = '/mnt/petrelfs/yangshuai1/rep/cogact_with_history/ckpt/Eagle2-2B'
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"

    tokenizer = AutoTokenizer.from_pretrained(llm_backbone_id, use_fast=True)

    processor = EagleProcessor(
        llm_backbone_id,
        max_input_tiles=1,
        model_spec=SimpleNamespace(
            num_image_token = 256,
            template = "qwen2-chat"
        ),
    )

    llm_backbone = AutoModel.from_pretrained(
        llm_backbone_id,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
        )
    
    llm_backbone.img_context_token_id = processor.get_img_context_token()
    assert llm_backbone.template == processor.model_spec.template
    assert llm_backbone.num_image_token == processor.model_spec.num_image_token

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{llm_backbone_id}[/] from Checkpoint")
    vlm = CogACT(
        vlm = llm_backbone,
        config_json = config_json,
        tokenizer = tokenizer,
        processor = processor,
        token_size= llm_backbone.config.llm_config.hidden_size
    )

    return vlm

# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    model_type: str = "pretrained",
    **kwargs,
) -> CogACT:
    """Loads a pretrained CogACT from either local disk or the HuggingFace Hub."""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(model_id_or_path)))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/*.pt")
        if (len(valid_ckpts) == 0) or (len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        target_ckpt = Path(valid_ckpts[-1]).name
        model_id_or_path = str(model_id_or_path)  # Convert to string for HF Hub API
        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            # relpath = Path(model_type) / model_id_or_path
            config_json = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{('config.json')!s}", cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{('dataset_statistics.json')!s}", cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{(Path('checkpoints') / target_ckpt)!s}", cache_dir=cache_dir
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    vla = CogACT.from_pretrained(
        checkpoint_pt,
        freeze_weights=not load_for_training,
        norm_stats = norm_stats,
        **kwargs,
    )

    return vla


