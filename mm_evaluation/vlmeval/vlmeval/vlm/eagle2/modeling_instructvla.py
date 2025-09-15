"""
modeling_instructvla.py

This file provide a simple loading function to load the **vlm part** of the InstructVLA

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

from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
import transformers
from packaging import version

import os
import json

import time
import timm
import random

from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F


import math
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

from .eagle_utils import EagleProcessor, extract_decoder_hidden_states, cleanup_xlora_pre_hooks, model_forward
from types import SimpleNamespace
from transformers import StoppingCriteria, StoppingCriteriaList

# Initialize Overwatch =>> Wraps `logging.Logger`

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# This system message is JUST a placeholder
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

class InstructVLA(nn.Module):
    def __init__(
        self,
        vlm: AutoModel,
        processor: AutoProcessor = None,
        tokenizer: AutoTokenizer = None,
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 1,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        config_json = None,
        meta_token_ids = None,
        stage = "stage1",
        **kwargs,
    ) -> None:
        super().__init__()
        self.action_model = None
        
        self.vlm = vlm
        self.processor = processor
        self.tokenizer = tokenizer
        self.config_json = config_json
        self.token_size = token_size
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size

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
        self.vlm.language_model.forward = model_forward(self.vlm.language_model)
        self.vlm.language_model.transformer_layer_cls = Qwen2DecoderLayer
        

        self.vlm.neftune_alpha = None
        self.action_dim = action_dim
        self.meta_token_ids = meta_token_ids
        self.min_meta_token = self.meta_token_ids[0]
        self.max_meta_token = self.meta_token_ids[-1]

        # Freeze all parameters in the model
        if stage == "stage1":
            print("Train the model in stage 1 with lora and learnable embeddings")
            for param in self.vlm.parameters():
                param.requires_grad = False

            lora_config = LoraConfig(
                    r=128,
                    lora_alpha=256,
                    target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 
                                    'self_attn.o_proj', 'mlp.down_proj', 'mlp.up_proj'],  # adjust based on model architecture
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM  # assuming a causal language model
                )
            
            # Unfreeze only the new tokens' embeddings
            
            self.vlm.language_model = get_peft_model(self.vlm.language_model, lora_config)

            new_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.new_tokens)  # Convert new tokens to their respective ids
            new_token_ids = torch.tensor(new_token_ids, device=self.vlm.language_model.base_model.model.lm_head.weight.device)

            # we unfreeze the embed and lm_head, but not all parameters are finetuned, the trainable weight are less than the reported weight
            self.vlm.language_model.base_model.model.lm_head.weight.requires_grad = True
            self.vlm.language_model.base_model.model.model.embed_tokens.weight.requires_grad = True

            def mask_old_token_grad(grad: torch.Tensor):
                mask = torch.ones(grad.shape[0], dtype=torch.bool, device=grad.device)
                mask[new_token_ids] = False 
                grad[mask] = 0
                return grad

            self.vlm.language_model.base_model.model.lm_head.weight.register_hook(mask_old_token_grad)
            self.vlm.language_model.base_model.model.model.embed_tokens.weight.register_hook(mask_old_token_grad)


        elif stage == "stage2":
            print("Train the model in stage 2 with X-LoRA")

            # We defaultly use the dense method, the inital scale of each adapter is 1/num_adapter
            # The language expert is initaled from zero
            from peft import XLoraConfig

            empty_language_adapter = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "ckpt", "empty_language_adapter")
            )

            lora_config = XLoraConfig(
                task_type="CAUSAL_LM",
                hidden_size=token_size,
                xlora_depth=4,
                xlora_size=128,
                adapters={
                    "0": empty_language_adapter,
                    "1": empty_language_adapter, # Must be replaced if you want to train the stage-2 model !
                },
            )

            print(
                f"If you are initializing stage-2 training from a stage-1 model, "
                f"you **must** replace **one of** the above empty_language_adapter with "
                f"the unloaded LoRA module from your stage-1 model. "
                f"Because the unloaded stage-1.pt does not contain the pretrained action "
                f"LoRA expert weights."
            )

            self.vlm.language_model = get_peft_model(self.vlm.language_model, lora_config)

            new_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.new_tokens)

            new_token_ids = torch.tensor(new_token_ids, device=self.vlm.language_model.base_model.model.lm_head.weight.device)
            # we unfreeze the embed and lm_head, but not all parameters are finetuned, the trainable weight are less than the reported weight
            self.vlm.language_model.base_model.model.lm_head.weight.requires_grad = True
            self.vlm.language_model.base_model.model.model.embed_tokens.weight.requires_grad = True

            def mask_old_token_grad(grad: torch.Tensor):
                mask = torch.ones(grad.shape[0], dtype=torch.bool, device=grad.device)
                mask[new_token_ids] = False 
                grad[mask] = 0
                return grad

            self.vlm.language_model.base_model.model.lm_head.weight.register_hook(mask_old_token_grad)
            self.vlm.language_model.base_model.model.model.embed_tokens.weight.register_hook(mask_old_token_grad)

            for name, param in self.vlm.language_model.base_model.lora_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.stage = stage

        self.vlm.language_model.print_trainable_parameters()

        self.vlm.language_model.transformer_layer_cls = Qwen2DecoderLayer

        for name, param in self.named_parameters():
            param.data = param.data.to(torch.float32)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(torch.float32)


        class StoppingCriteriaSub(StoppingCriteria):
            def __init__(self,tokenizer, stops = [], encounters=1):
                super().__init__()
                self.stops = [stop.to("cuda") for stop in stops]
                self.tokenizer = tokenizer

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                last_token = input_ids[0][-1]
                for stop in self.stops:
                    if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                        return True
                return False

        stop_words = ["<new_token_0>"]
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, tokenizer = self.tokenizer)])

        self._fix_system1 = False

        # for inference
        self.last_response = None
        self.run_index = 0
        self.latent = None
    
    @property
    def fix_system1(self):
        return self._fix_system1

    @fix_system1.setter
    def fix_system1(self, value: bool):
        print(f"fix_system1 is being updated to: {value}")
        if value:
            for name, param in self.action_model.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.action_model.named_parameters():
                param.requires_grad = True

        self._fix_system1 = value

    @property
    def llm_backbone(self):
        return self.vlm.language_model
    
    @property
    def vision_backbone(self):
        return self.vlm.vision_model
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)
    
    @torch.inference_mode()
    def chat(self, *args, **kwargs):
        # chat method from eagle vlm
        autocast_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
            ret = self.vlm.chat(*args, **kwargs)
        return ret

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        system1_pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        per_device_batch_size: int = 16,
        action_masks = None,
        image_flags = None,
        sampling_type = None,
        t = None,
        train_idx = 0,
        **kwargs,
    ) -> Tuple:
        if actions is None:
            # standard vlm forward
            # from IPython import embed;embed()
            per_device_batch_size = input_ids.shape[0]
            output: CausalLMOutputWithPast = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                image_flags=torch.ones((input_ids.shape[0],1)).to( device=input_ids.device) if image_flags is None else image_flags,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict=return_dict,
                fast_loss_cal=True,
                **kwargs,
            )
            return output.loss, output
        else:

            assert per_device_batch_size == actions.shape[0]

            output: CausalLMOutputWithPast = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                # inputs_embeds=inputs_embeds,
                image_flags=torch.ones((input_ids.shape[0],1)).to( device=input_ids.device) if image_flags is None else image_flags,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                fast_loss_cal=True
            )

            # extract the last hidden state
            last_hidden_states = output.hidden_states[-1]
            
            # extract the latent action
            meta_feature_mask = (input_ids >= self.min_meta_token) & (input_ids <= self.max_meta_token)
            meta_feature = last_hidden_states[torch.where(meta_feature_mask==1)].view(meta_feature_mask.size(0),-1 , last_hidden_states.shape[-1])

            # actions_history = actions[:,0:self.past_action_window_size,:]
            actions_future = actions[:, -(self.future_action_window_size+1):, :]
            _, _, action_dim = actions_future.shape
            
            loss = self.action_model( latent_action = meta_feature,
                        pixel_values = dict(dino = system1_pixel_values),
                        actions = actions_future,
                        t = t,
                    )

            return loss + output.loss, output.loss


    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 5,
        norm_stats = None,
        stage = "stage1",
        num_of_meta_query = 64,
        **kwargs,
    ) -> InstructVLA:

        llm_backbone_id = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..","..","..","..","..", "ckpt", "Eagle2-2B")
            )

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = AutoModel.from_pretrained(llm_backbone_id,
                                        attn_implementation="flash_attention_2",
                                        trust_remote_code=True)

        processor = EagleProcessor(
            llm_backbone_id,
            max_input_tiles=1,
            model_spec=SimpleNamespace(
                num_image_token = 256,
                template = "qwen2-chat"
            ),
        )

        tokenizer = AutoTokenizer.from_pretrained(llm_backbone_id, 
                                                  use_fast=True,
                                                  trust_remote_code=True)

        new_tokens = ['<new_token_{}>'.format(i) for i in range(num_of_meta_query)]  # Create 256 new token names
        print(f"add {len(new_tokens)} latent action tokens")
        tokenizer.add_tokens(new_tokens)  # Add them to the tokenizer
        tokenizer.new_tokens = new_tokens
        processor.tokenizer = tokenizer

        # Resize the model's token embeddings to match the new vocabulary size
        try:  # Skip mean resize feature in new version of transformers
            vlm.language_model.resize_token_embeddings(len(tokenizer), mean_resizing = False)
        except Exception as e:
            vlm.language_model.resize_token_embeddings(len(tokenizer))

        # Freeze all parameters in the model
        for param in vlm.parameters():
            param.requires_grad = False

        # Unfreeze only the new tokens' embeddings
        new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)  # Convert new tokens to their respective ids

        vlm.img_context_token_id = processor.get_img_context_token()
        assert vlm.template == processor.model_spec.template
        assert vlm.num_image_token == processor.model_spec.num_image_token

        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        is_a_lora_model = any('lora' in i for i in  model_state_dict['language_model'])
        if not is_a_lora_model:
            print("No LoRA parameters in the checkpoint, so we load weight before init LoRA")
            vlm.language_model.load_state_dict(model_state_dict["language_model"])

        # Initialize InstructVLA
        instruct_vla = InstructVLA(vlm,
                        processor = processor,
                        tokenizer = tokenizer,
                        token_size = vlm.config.llm_config.hidden_size,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        past_action_window_size = past_action_window_size,
                        norm_stats = norm_stats,
                        meta_token_ids = new_token_ids,
                        stage = stage,
                        )


        assert (
            "mlp1" in model_state_dict and "language_model" in model_state_dict and "vision_model" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector`, `language_model` AND `vision_model`"

        instruct_vla.vlm.mlp1.load_state_dict(model_state_dict["mlp1"])
        instruct_vla.vlm.vision_model.load_state_dict(model_state_dict["vision_model"],strict=False) # The ckpt contains extra head parameters that are not use the previous transformers, but recently, the head is completely removed

        if is_a_lora_model:
            print("LoRA parameters in the checkpoint, so we load weight after init LoRA")
            instruct_vla.vlm.language_model.load_state_dict(model_state_dict["language_model"])
        

        # Load ActionModel from Checkpoint
        # if "action_model" in model_state_dict:
        #     instruct_vla.action_model.load_state_dict(model_state_dict["action_model"], strict=False)
        # else:
        #     print("No Action Model found in the pretrained checkpoint. Initializing a new one.")

        print("Since language evaluation does not require action head, the parameters are not loaded !!!!")
        return instruct_vla       

    @torch.inference_mode()
    def predict_action(
        self, 
        image: Image, 
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        use_generate = True,
        cache_latent = False,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        # Build VLA Prompt
    
        autocast_dtype = torch.bfloat16
        pixel_values = None

        # Prepare Inputs
        if use_generate and self.stage=="stage2":
            if self.last_response is None or self.run_index % 20 == 0:
                prompt = [
                    {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                    {
                        "role": "user",
                        "content": f"What action should the robot take to {instruction}? First answer my question.",
                        "image": [{'np_array': np.asarray(image)}],
                    },
                    {
                        "role": "assistant", 
                        "content": ""
                    }
                ]
                inputs = self.processor.prepare_input({"prompt": prompt})
                input_ids = inputs['input_ids'].to(self.vlm.device)
                pixel_values = inputs['pixel_values']

                pixel_values = pixel_values.to(self.vlm.device, dtype=autocast_dtype)
                
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
                    attention_mask = input_ids.ne(-10)
                    output: CausalLMOutputWithPast = self.vlm.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        max_new_tokens=50,
                        # fast_loss_cal=False,
                        output_hidden_states=False,
                        return_dict_in_generate=False,
                        stopping_criteria=self.stopping_criteria # to accelerate primitive
                    )

                # Extract cognition feature
                response = self.tokenizer.decode(output[0]).replace("<new_token_0>","")
                print("====== Reasoning ======= >" + response)
                self.last_response = response
            else:
                response = self.last_response

            prompt = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": f"What action should the robot take to {instruction}? First answer my question.",
                    "image": [{'np_array': np.asarray(image)}],
                },
                {
                    "role": "assistant", 
                    "content": response + " " + "".join(self.processor.tokenizer.new_tokens)
                }
            ]
        else:
            prompt = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": f"What action should the robot take to {instruction}?",
                    "image": [{'np_array': np.asarray(image)}],
                },
                {
                    "role": "assistant", 
                    "content": "".join(self.processor.tokenizer.new_tokens)
                }
            ]
        inputs = self.processor.preprocess_inputs_and_labels({"prompt": prompt})
        input_ids = inputs['input_ids'].to(self.vlm.device).unsqueeze(0)
        if pixel_values is None:
            # Preprocess Image
            pixel_values = inputs['pixel_values']
            pixel_values = pixel_values.to(self.vlm.device, dtype=autocast_dtype)

        # Generate cognition feature through vlm
        if cache_latent is False or (cache_latent is True and (self.run_index % 2 == 0 or self.latent is None)):
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
                )

            # Extract cognition feature
            last_hidden_states = output.hidden_states[-1]
            self.latent = last_hidden_states.detach()
        else:
            last_hidden_states = self.latent
            
        meta_feature_mask = (input_ids >= self.min_meta_token) & (input_ids <= self.max_meta_token)
        meta_feature = last_hidden_states[torch.where(meta_feature_mask==1)].view(meta_feature_mask.size(0),-1 , last_hidden_states.shape[-1])

        BS, step, dim = meta_feature.shape

        sys1_pixel_values = dict(dino = self.action_model.default_dino_transform(image).unsqueeze(0).to(self.vlm.device))
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
            samples = self.action_model.sampling(   latent_action = meta_feature,
                                                    pixel_values = sys1_pixel_values,
                                                    )
        normalized_actions = samples[0].float().cpu().numpy()

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
        self.run_index += 1
        cleanup_xlora_pre_hooks(self.vlm.language_model)
        return actions, normalized_actions, meta_feature

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


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    llm_backbone_id=None,
    stage = 'stage1',
    num_of_meta_query = 64,
):

    if llm_backbone_id is None:
        llm_backbone_id = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ckpt", "Eagle2-2B")
        )

    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        print(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

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

    vlm = AutoModel.from_pretrained(
        llm_backbone_id,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
        )

    new_tokens = ['<new_token_{}>'.format(i) for i in range(num_of_meta_query)]  # Create 256 new token names
    print(f"add {len(new_tokens)} latent action tokens")
    tokenizer.add_tokens(new_tokens)  # Add them to the tokenizer
    tokenizer.new_tokens = new_tokens
    processor.tokenizer = tokenizer

    # Resize the model's token embeddings to match the new vocabulary size
    try:  # Skip mean resize feature in new version of transformers
        vlm.language_model.resize_token_embeddings(len(tokenizer), mean_resizing = False)
    except Exception as e:
        vlm.language_model.resize_token_embeddings(len(tokenizer))

    vlm.img_context_token_id = processor.get_img_context_token()
    assert vlm.template == processor.model_spec.template
    assert vlm.num_image_token == processor.model_spec.num_image_token

    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    print(f"Loading VLM [bold blue]{llm_backbone_id}[/] from Checkpoint")
    vla = InstructVLA(
        vlm = vlm,
        config_json = config_json,
        tokenizer = tokenizer,
        processor = processor,
        token_size= vlm.config.llm_config.hidden_size,
        meta_token_ids = new_token_ids,
        stage = stage
    )

    return vla

# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    model_type: str = "pretrained",
    **kwargs,
) -> InstructVLA:
    """Loads a pretrained InstructVLA from either local disk or the HuggingFace Hub."""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        print(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

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
        print(f"Checking HF for `{(hf_path := str(Path(model_id_or_path)))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/*.pt")
        if (len(valid_ckpts) == 0) or (len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        target_ckpt = Path(valid_ckpts[-1]).name
        model_id_or_path = str(model_id_or_path)  # Convert to string for HF Hub API
        print(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
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
    with open(config_json, "r") as f:
        config = json.load(f)
    num_of_meta_query, num_of_meta_query_from_config = None, None
    if 'num_of_meta_query' in kwargs:
        num_of_meta_query = kwargs.pop("num_of_meta_query")
    if 'num_of_meta_query' in config:
        num_of_meta_query_from_config = config.pop("num_of_meta_query")

    if num_of_meta_query is not None and num_of_meta_query_from_config is not None:
        assert num_of_meta_query == num_of_meta_query_from_config, f'you need {num_of_meta_query} meta queries, but the checkpoint is trained with {num_of_meta_query} meta queries.'
    elif num_of_meta_query is None and num_of_meta_query_from_config is None:
        num_of_meta_query = 64
        print(f"not specify the number of meta query, the default value is 64 !!")

    if num_of_meta_query_from_config is not None:
        num_of_meta_query = num_of_meta_query_from_config

    if 'stage' in config:
        training_stage_from_config = config.pop("stage")
    if 'stage' in kwargs:
        _ = kwargs.pop("stage")
    
    vla = InstructVLA.from_pretrained(
        checkpoint_pt,
        freeze_weights=not load_for_training,
        norm_stats = norm_stats,
        num_of_meta_query=num_of_meta_query,
        stage = training_stage_from_config,
        **kwargs,
    )

    return vla


