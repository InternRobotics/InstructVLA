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
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer


import time
import timm
import random

from peft import get_peft_model, LoraConfig, TaskType, XLoraConfig
import torch.nn.functional as F

import math
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from vla.eagle_utils import EagleProcessor, extract_decoder_hidden_states
from types import SimpleNamespace
from transformers import StoppingCriteria, StoppingCriteriaList

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant. Please help me control the robot."


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction="mean")
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


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
        # forward_lm_head: bool = False, # removed !
        fast_loss_cal: bool = False,
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
        loss, logits = None, None

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if not fast_loss_cal: # Keep original loss logic
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        else: # since the qwen model has a vary large lm head, we only calculate the necessary part
            logits = None
            ignore_index = -100
            labels_padded = nn.functional.pad(labels, (0, 1), value=ignore_index)   # (B, L+1)
            shift_labels = labels_padded[..., 1:].contiguous()                      # (B, L)

            keep_mask = shift_labels.ne(ignore_index)                               # bool, (B, L)

            if  keep_mask.any():
                # hidden_states: (B, L, H)  -->  (N_keep, V)
                logits = self.lm_head(hidden_states[keep_mask])

            loss = ForCausalLMLoss(
                logits=logits,                               # (N_keep, V) or None
                labels=None,                                 # unused when shift_labels passed
                shift_labels=shift_labels[keep_mask],        # 1‑D tensor with no -100 values
                vocab_size=self.config.vocab_size,
                **kwargs,
            )


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

from .film_vit import FiLMedDinoVisionBackbone

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.half_dim = dim // 2
        self.max_period = max_period

    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        emb = math.log(self.max_period) / (self.half_dim - 1)
        emb = torch.exp(
            torch.arange(self.half_dim, device=t.device, dtype=t.dtype) * -emb
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEncoder(nn.Module):
    def __init__(self,
                head_token_size,
                action_dim
                ):
        super().__init__()
        
        self.linear_1 =nn.Linear(action_dim, head_token_size, bias=True)
        self.linear_2 = nn.Linear(2 * head_token_size, head_token_size)
        self.nonlinearity = nn.SiLU()
        self.linear_3 = nn.Linear(head_token_size, head_token_size)
    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        emb = self.linear_1(action)
        time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
        emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb

class ActionWorldModel(nn.Module):
    def __init__(self, 
                 token_size,
                 past_action_window_size,
                 future_action_window_size,
                 action_dim,
                 VQTokenizer = None,
                 VideoGPT = None,
                 world_model = '/mnt/petrelfs/yangshuai1/rep/TMP_CogACTmini_x_DIT_Atten_HisF_MultiF_R_Silence/world_model/ivideogpt-oxe-256-act-free',
                 pertrained_dino = '/mnt/petrelfs/yangshuai1/rep/cogact_with_history/ckpt/dinov2-oxe/pytorch_model.bin',
                 head_token_size = 768,
                 default_image_size = 224,
                 ):
        super().__init__()
        self.action_head = nn.Sequential(  nn.Linear(head_token_size * (future_action_window_size + 1), head_token_size*4, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(head_token_size*4, head_token_size*2, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(head_token_size*2, head_token_size//2, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(head_token_size//2, (future_action_window_size + 1) * action_dim, bias=True),
                                        ) # 120M
        vision_model = timm.create_model(  'vit_large_patch14_reg4_dinov2.lvd142m',
                                                pretrained=True,
                                                num_classes=0,  # remove classifier nn.Linear
                                                img_size = 224
                                            )
        # pretrained dino from OpenVLA
        # vision_model.load_state_dict(torch.load(pertrained_dino))
        self.film_vision_model = FiLMedDinoVisionBackbone(vision_model, token_size)

        self.visual_projector = nn.Sequential(  nn.Linear(1024, head_token_size, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(head_token_size, head_token_size, bias=True),
                                                )

        self.cog_projector = nn.Sequential( nn.Linear(token_size, head_token_size, bias=True),
                                            nn.SiLU(),
                                            nn.Linear(head_token_size, head_token_size, bias=True),
                                            )
        
        self.default_image_size = default_image_size
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.film_vision_model)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        self.default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        
        self.future_queries = nn.Parameter(torch.randn(future_action_window_size + 1, head_token_size))
        self.future_action_window_size = future_action_window_size
        self.action_dim = action_dim
        
        if VQTokenizer is None or VideoGPT is None:
            print("Initialize world model from config (no pretrained weights)")
            # self.VQTokenizer = CompressiveVQModel.from_pretrained(world_model, subfolder='tokenizer', low_cpu_mem_usage=False)
            transformer_config = AutoConfig.from_pretrained(world_model, subfolder='transformer')
            self.VideoGPT = AutoModelForCausalLM.from_config(transformer_config, attn_implementation="eager")
        else:
            # self.VQTokenizer = VQTokenizer
            self.VideoGPT = VideoGPT

        # remove unused parameters
        if hasattr(self.VideoGPT, 'lm_head'):
            self.VideoGPT.lm_head = nn.Identity()

        if hasattr(self.VideoGPT.model, 'embed_tokens'):
            self.VideoGPT.model.embed_tokens = nn.Identity()

        self.flow_t_max = 0.999
        self.flow_sig_min = 0.001
        self.flow_beta_dist = torch.distributions.Beta(1.5, 1)
        self.flow_sampling = "beta"
        self.time_embedding = SinusoidalPosEmb(
            head_token_size, 100.0
        )
        self.action_embed = ActionEncoder( head_token_size=head_token_size,
                                            action_dim=action_dim
                                        )
        
    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t
    
    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1
    
            
    def forward(self,
                cognition_features: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                actions: Optional[torch.FloatTensor] = None,
                t: Optional[torch.FloatTensor] = None,
                indices_for_now=None):

        batch_size = cognition_features.shape[0]
        if indices_for_now is not None:
            pixel_values = pixel_values['dino'][indices_for_now]
        else:
            pixel_values = pixel_values['dino']
        
        visual_embed = self.film_vision_model(pixel_values, cognition_features)

        visual_feature = self.visual_projector(visual_embed)
        cognition_features = self.cog_projector(cognition_features)

        # prepare flow matching variables

        # t = self.sample_fm_time(actions.shape[0]).to( device=actions.device, dtype=actions.dtype)
        x0 = torch.randn_like(actions, device=actions.device, dtype=actions.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        time_cond = self.time_embedding(t)
        action_embeds = self.action_embed(psi_t, time_cond)

        # prepare action queries
        input_seq = torch.cat([visual_feature, cognition_features, action_embeds], dim=1)

        vis_len = visual_feature.shape[1]
        cog_len = cognition_features.shape[1]
        fut_len = action_embeds.shape[1]

        # make 4d mask
        total_len = vis_len + cog_len + fut_len

        # blockwise causal attention mask
        attention_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=input_seq.device)
        attention_mask[:vis_len, :vis_len] = 1
        attention_mask[vis_len:vis_len+cog_len, :vis_len+cog_len] = 1
        attention_mask[vis_len+cog_len:, :total_len] = 1

        # expand to 4d mask
        # [BS, 1, q_len, kv_len]
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, total_len, total_len)

        encoded_seq = self.VideoGPT.model(
            inputs_embeds=input_seq,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
    
        future_pred = encoded_seq.hidden_states[-1][:, -(self.future_action_window_size + 1):]
        future_pred = future_pred.reshape(batch_size, -1)

        output = self.action_head(future_pred)
        v_psi  = output.reshape(batch_size, self.future_action_window_size + 1, self.action_dim)
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)

    def sampling(self,
                cognition_features: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                num_inference_steps: int = 10,
                ):
        # prepare features
        pixel_values = pixel_values['dino']
        device = cognition_features.device
        dtype = cognition_features.dtype

        # from IPython import embed;embed()
        visual_embed = self.film_vision_model(pixel_values, cognition_features)

        visual_feature = self.visual_projector(visual_embed)
        cognition_features = self.cog_projector(cognition_features)

        # sample pure action noise
        actions = torch.randn(
            (1, self.future_action_window_size+1, self.action_dim), device=device, dtype=dtype
        )

        delta_t = 1.0 / num_inference_steps
        t = torch.zeros(1, device=device, dtype=dtype)

        attention_mask = None

        for _ in range(num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            action_embeds = self.action_embed(actions, time_cond)

            input_seq = torch.cat([visual_feature, cognition_features, action_embeds], dim=1)

            if attention_mask is None:
                vis_len = visual_feature.shape[1]
                cog_len = cognition_features.shape[1]
                fut_len = action_embeds.shape[1]

                # make 4d mask
                total_len = vis_len + cog_len + fut_len

                # blockwise causal attention mask
                attention_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=input_seq.device)
                attention_mask[:vis_len, :vis_len] = 1
                attention_mask[vis_len:vis_len+cog_len, :vis_len+cog_len] = 1
                attention_mask[vis_len+cog_len:, :total_len] = 1

                # expand to 4d mask
                # [BS, 1, q_len, kv_len]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(1).expand(1, 1, total_len, total_len)
            
            encoded_seq = self.VideoGPT(
                inputs_embeds=input_seq,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )

            future_pred = encoded_seq.hidden_states[-1][:, -(self.future_action_window_size + 1):]
            action_embeds = future_pred.reshape(1, -1)

            action_vel = self.action_head(action_embeds).reshape(1, self.future_action_window_size + 1, self.action_dim)
            actions += delta_t * action_vel
            t += delta_t

        return actions



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
        meta_token_ids = None,
        stage = "stage1",
        **kwargs,
    ) -> None:
        super().__init__()
        self.action_model = ActionWorldModel(token_size,past_action_window_size,future_action_window_size,action_dim)
        
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
        self.vlm.language_model.forward = model_forward(self.vlm.language_model)
        self.vlm.language_model.transformer_layer_cls = Qwen2DecoderLayer
        

        self.vlm.neftune_alpha = None
        self.action_dim = action_dim
        self.meta_token_ids = meta_token_ids
        self.min_meta_token = self.meta_token_ids[0]
        self.max_meta_token = self.meta_token_ids[-1]

        # Freeze all parameters in the model
        if stage == "stage1":
            overwatch.info("Train the model in stage 1 with lora and learnable embeddings")
            for param in self.vlm.parameters():
                param.requires_grad = False

            lora_config = LoraConfig(
                    r=128,
                    lora_alpha=256,
                    target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.down_proj', 'mlp.up_proj'],  # adjust based on model architecture
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM  # assuming a causal language model
                )
            
            # Unfreeze only the new tokens' embeddings
            
            self.vlm.language_model = get_peft_model(self.vlm.language_model, lora_config)

            new_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.new_tokens)  # Convert new tokens to their respective ids
            for token_id in new_token_ids:
                self.vlm.language_model.base_model.model.model.embed_tokens.weight[token_id].requires_grad = True  # Unfreeze new token embeddings
                self.vlm.language_model.base_model.model.lm_head.weight[token_id].requires_grad = True
        elif stage == "stage2":
            overwatch.info("Train the model in stage 2 with X-LoRA")
            
            lora_config = XLoraConfig(
                task_type="CAUSAL_LM",
                hidden_size=token_size,
                xlora_depth=4,
                xlora_size=128,
                adapters={
                    "0": "/mnt/petrelfs/yangshuai1/rep/cogact_with_history/outputs/head_balation/sys12_meta_query_action_only_sync_pretraining_v2_query_64_mlp_lora--image_aug/checkpoints/empty",
                    "1": "/mnt/petrelfs/yangshuai1/rep/cogact_with_history/outputs/head_balation/sys12_meta_query_action_only_sync_pretraining_v2_query_64_mlp_lora--image_aug/checkpoints/step-036000-epoch-09_lora_only",
                },
            )
            self.vlm.language_model = get_peft_model(self.vlm.language_model, lora_config)

            new_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.new_tokens)  # Convert new tokens to their respective ids
            for token_id in new_token_ids:
                self.vlm.language_model.base_model.lora_model.model.model.embed_tokens.weight[token_id].requires_grad = True  # Unfreeze new token embeddings
                self.vlm.language_model.base_model.lora_model.model.lm_head.weight[token_id].requires_grad = True

            for name, param in self.vlm.language_model.base_model.lora_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

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


    @property
    def llm_backbone(self):
        return self.vlm.language_model
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_model
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        system2_pixel_values: Optional[torch.FloatTensor] = None,
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
        image_flags = None,
        sampling_type = None,
        t = None,
        train_idx = 0,
        **kwargs,
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        '''
        VLM Input & Repeated Cognition Features
        (Index range: 0 to 13 for B=2, N+1=7)

        VLM Input Index:         0   1   2   3   4   5   6     7   8   9  10  11  12  13
        Sample Index:           [0   0   0   0   0   0   0]   [1   1   1   1   1   1   1]
        Timestep in Sample:      0   1   2   3   4   5   6     0   1   2   3   4   5   6
        Shift Offset:            6   5   4   3   2   1   0     6   5   4   3   2   1   0
                                 ↑    past frames    ↑   *     ↑    past frames    ↑   *
                        (used to slice action target windows from right end of trajectory)

        repeated_cognition_features: (used in action_model)
                                └──────── sample 0 ───────┘    └─────── sample 1 ───────┘

        action_model pred[i] corresponds to:
        → future_actions[i] = actions[batch_idx, -(T+shift_offset):-shift_offset]
        vlm takes the first frames and head takes all frames
        '''

        # from IPython import embed;embed()
        # indices_for_past = torch.cat([torch.arange(i*(self.past_action_window_size+1), i*(self.past_action_window_size+1)+self.past_action_window_size) for i in range(per_device_batch_size)])
        # indices_for_now = torch.arange(self.past_action_window_size, per_device_batch_size*(self.past_action_window_size+1), self.past_action_window_size+1)

        if actions is None:
            # make it a vlm
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
            assert input_ids.shape[0] == (per_device_batch_size * (self.past_action_window_size+1)), f"input_ids.shape[0] = {input_ids.shape[0]}, but should be {per_device_batch_size * (self.past_action_window_size+1)} "

            first_past_indices = torch.arange(per_device_batch_size) * (self.past_action_window_size + 1)

            # from IPython import embed;embed()

            output: CausalLMOutputWithPast = self.vlm(
                input_ids=input_ids[first_past_indices],
                attention_mask=attention_mask[first_past_indices],
                pixel_values=pixel_values[first_past_indices],
                labels=labels[first_past_indices],
                # inputs_embeds=inputs_embeds,
                image_flags=torch.ones((first_past_indices.shape[0],1)).to( device=input_ids.device) if image_flags is None else image_flags,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                fast_loss_cal=True
            )

            # extract the last hidden state and the learnable EOS token feature
            last_hidden_states = output.hidden_states[-1]
            
            meta_feature_mask = (input_ids[first_past_indices] >= self.min_meta_token) & (input_ids[first_past_indices] <= self.max_meta_token)
            meta_feature = last_hidden_states[torch.where(meta_feature_mask==1)].view(meta_feature_mask.size(0),-1 , last_hidden_states.shape[-1])
            # extract the cognition feature

            repeated_cognition_features = meta_feature.repeat_interleave(self.past_action_window_size + 1, dim=0)
            assert repeated_cognition_features.shape[0] == per_device_batch_size * (self.past_action_window_size+1)

            # actions_history = actions[:,0:self.past_action_window_size,:]
            shift_offsets = torch.tensor( list(range(self.past_action_window_size, -1, -1)) * per_device_batch_size )
            batch_indices = torch.repeat_interleave(torch.arange(per_device_batch_size), self.past_action_window_size + 1)
            actions_future = torch.stack([
                actions[batch_idx, -(self.future_action_window_size + 1 + offset):-offset if offset > 0 else None, :]
                for batch_idx, offset in zip(batch_indices, shift_offsets)
            ], dim=0)

            _, _, action_dim = actions_future.shape

            # do batch sample
            # k = 2 # number of samples 
            t =  t.repeat(self.past_action_window_size + 1)
            if sampling_type is None:
                loss = self.action_model( cognition_features = repeated_cognition_features,
                            pixel_values = dict(dino = system2_pixel_values),
                            actions = actions_future,
                            t = t,
                        )
            else:
                B = per_device_batch_size
                W = self.past_action_window_size + 1

                sampled_indices = []
                if sampling_type == 'first':
                    for b in range(B):
                        base = b * W
                        sampled_indices.append(base + torch.ones(1, dtype=torch.long, device=repeated_cognition_features.device))
                elif sampling_type == 'random':
                    for b in range(B):
                        base = b * W
                        idx = torch.randperm(W, device=repeated_cognition_features.device)[:k]
                        sampled_indices.append(base + idx)
                
                sampled_indices = torch.cat(sampled_indices, dim=0)
                loss = self.action_model( cognition_features = repeated_cognition_features[sampled_indices],
                    pixel_values = dict(dino = system2_pixel_values[sampled_indices]),
                    actions = actions_future[sampled_indices],
                    t = t[sampled_indices],
                )

            return loss + output.loss, output.loss

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT

        vit_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={SiglipEncoderLayer})
        transformer_block_policy_1 = partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer})
        transformer_block_policy_2 = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})

        from transformers.models.llama.modeling_llama import LlamaMLP
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, Qwen2MLP, LlamaMLP},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vit_wrap_policy,
                transformer_block_policy_1,
                transformer_block_policy_2,
                prismatic_fsdp_wrapping_policy,
            ],
        )

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
        stage = "stage1",
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

        new_tokens = ['<new_token_{}>'.format(i) for i in range(64)]  # Create 256 new token names
        tokenizer.add_tokens(new_tokens)  # Add them to the tokenizer
        tokenizer.new_tokens = new_tokens
        processor.tokenizer = tokenizer

        # Resize the model's token embeddings to match the new vocabulary size
        vlm.language_model.resize_token_embeddings(len(tokenizer))  # Resize the model's embeddings

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
            overwatch.warning("No LoRA parameters in the checkpoint, so we load weight before init LoRA")
            vlm.language_model.load_state_dict(model_state_dict["language_model"])

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
                        meta_token_ids = new_token_ids,
                        stage = stage,
                        )


        assert (
            "mlp1" in model_state_dict and "language_model" in model_state_dict and "vision_model" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector`, `language_model` AND `vision_model`"

        cogact.vlm.mlp1.load_state_dict(model_state_dict["mlp1"])
        cogact.vlm.vision_model.load_state_dict(model_state_dict["vision_model"],strict=False) # The ckpt contains extra head parameters that are not use the previous transformers, but recently, the head is completely removed

        if is_a_lora_model:
            overwatch.warning("LoRA parameters in the checkpoint, so we load weight after init LoRA")
            cogact.vlm.language_model.load_state_dict(model_state_dict["language_model"])
        

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            cogact.action_model.load_state_dict(model_state_dict["action_model"])
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
        use_generate = False,
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
    
        autocast_dtype = torch.bfloat16
        pixel_values = None

        # Prepare Inputs
        if use_generate:

            prompt = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": f"What action should the robot take to {instruction}? Give both move primitive and action.",
                    "image": [{'np_array': np.asarray(image)}],
                },
                {
                    "role": "assistant", 
                    "content": "".join(self.processor.tokenizer.new_tokens)
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
                    max_length=1024,
                    # fast_loss_cal=False,
                    output_hidden_states=False,
                    return_dict_in_generate=False,
                    stopping_criteria=self.stopping_criteria
                )

            # Extract cognition feature
            primitive = self.tokenizer.decode(output[0]).replace("<new_token_0>","")
            print(primitive)

            prompt = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": f"What action should the robot take to {instruction}? Give both move primitive and action.",
                    "image": [{'np_array': np.asarray(image)}],
                },
                {
                    "role": "assistant", 
                    "content": primitive +"".join(self.processor.tokenizer.new_tokens)
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
            pixel_values = pixel_values.to(self.vlm.device)

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
        last_hidden_states = output.hidden_states[-1]
            
        meta_feature_mask = (input_ids >= self.min_meta_token) & (input_ids <= self.max_meta_token)
        meta_feature = last_hidden_states[torch.where(meta_feature_mask==1)].view(meta_feature_mask.size(0),-1 , last_hidden_states.shape[-1])
        # Sample random noise
        BS, step, dim = meta_feature.shape

        sys1_pixel_values = dict(dino = self.action_model.default_dino_transform(image).unsqueeze(0).to(self.vlm.device))
        samples = self.action_model.sampling(   cognition_features = meta_feature,
                                                pixel_values = sys1_pixel_values,
                                                )
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
    image_processor: AutoProcessor

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        
        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        img = rlds_batch["observation"]["image_primary"]

        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        pixel_values = []
        system2_pixel_values = []
        input_ids = []
        labels = []
        # print(rlds_batch)

        anno = json.loads(rlds_batch["reasonings"].decode())
        move_primitive = anno["move_primitive"]
        # tokenizer = self.processor.tokenizer
        for i in img:
            if random.random() < 0.2:
                action_prompt = f"What action should the robot take to {lang}? Give both move primitive and action."
                assistant_content = f"{move_primitive} " + "".join(self.processor.tokenizer.new_tokens)
            else:
                action_prompt = f"What action should the robot take to {lang}?"
                assistant_content = "".join(self.processor.tokenizer.new_tokens)

            prompt = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": action_prompt,
                    "image": [{'np_array': i}],
                },
                {
                    "role": "assistant", 
                    "content": assistant_content
                }
            ]

            inputs = self.processor.preprocess_inputs_and_labels({"prompt": prompt})

            pixel_values.append(inputs['pixel_values'])
            img_array = np.squeeze(i).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            system2_pixel_values.append(self.image_processor(pil_img))

            input_ids.append(inputs['input_ids'].unsqueeze(0))
            labels.append(inputs['labels'].unsqueeze(0))

        pixel_values = torch.stack(pixel_values)
        system2_pixel_values = torch.stack(system2_pixel_values)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)

        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, 
                    dataset_name=dataset_name, actions=action, action_masks=action_mask, 
                    episode_idx=rlds_batch["idx"], frame_idx=rlds_batch['frame_idx'],
                    system2_pixel_values=system2_pixel_values,
                    )

from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from itertools import cycle

@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    mm_dataloader: DataLoader = None

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        batch_pixel_values   = [inst["pixel_values"]            for inst in instances]  # List[Tensor], each [M_i, C, H, W]
        system2_pixel_values = [inst["system2_pixel_values"]    for inst in instances]  # List[Tensor], each [M_i, C, H, W]
        batch_input_ids      = [inst["input_ids"]               for inst in instances]  # List[Tensor], each [M_i, seq_len]
        batch_labels         = [inst["labels"]                  for inst in instances]  


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
        system2_pixel_values = torch.cat(system2_pixel_values, dim=0).to(self.pixel_values_dtype).squeeze(1)
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
            system2_pixel_values=system2_pixel_values,
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

        if self.mm_dataloader is not None:
            # co-training VLM
            mm_batch = next(self.mm_dataloader)
        else:
            mm_batch = None
        return output, mm_batch

def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    tokenizer: PreTrainedTokenizerBase,
    processor: AutoProcessor,
    image_processor: AutoProcessor,
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,         # Concatenated `past_action_window_size-1' actions and the current action for the input
    load_all_data_for_training: bool = True,  # Load all data for training, or only a subset
    base_action_tokenizer: PreTrainedTokenizerBase = None,
    mm_dataset = None,
    mm_collator = None,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    if base_action_tokenizer is None:
        action_tokenizer = None
    else:
        action_tokenizer = ActionTokenizer(base_action_tokenizer)
    # action_tokenizer = ActionTokenizer(tokenizer)
    if mm_dataset is not None:
        sampler = DistributedSampler(
            mm_dataset,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            shuffle=True,
            seed=42,
            drop_last=True,
        )

        mm_dataloader = DataLoader(
            mm_dataset,
            batch_size=2,
            sampler=sampler,
            collate_fn=mm_collator,
            num_workers=4,
        )
    mm_iter = iter(mm_dataloader) if mm_dataset is not None else None
    mm_iter = cycle(mm_iter) if mm_iter is not None else None # make it infinite

    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, processor, image_processor
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side, mm_dataloader = mm_iter
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
    llm_backbone_id = '/mnt/petrelfs/yangshuai1/rep/cogact_with_history/ckpt/Eagle2-2B',
    stage = 'stage1',
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

    vlm = AutoModel.from_pretrained(
        llm_backbone_id,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
        )

    new_tokens = ['<new_token_{}>'.format(i) for i in range(64)]  # Create 256 new token names
    tokenizer.add_tokens(new_tokens)  # Add them to the tokenizer
    tokenizer.new_tokens = new_tokens
    processor.tokenizer = tokenizer

    # Resize the model's token embeddings to match the new vocabulary size
    vlm.language_model.resize_token_embeddings(len(tokenizer))  # Resize the model's embeddings

    vlm.img_context_token_id = processor.get_img_context_token()
    assert vlm.template == processor.model_spec.template
    assert vlm.num_image_token == processor.model_spec.num_image_token

    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{llm_backbone_id}[/] from Checkpoint")
    vla = CogACT(
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


