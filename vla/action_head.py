import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import timm
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any, Sequence
import os
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

class ActionModel(nn.Module):
    def __init__(self, 
                 token_size,
                 past_action_window_size,
                 future_action_window_size,
                 action_dim,
                 expert_backbone=None,
                 default_image_size=224,
                 ):
        super().__init__()

        if expert_backbone is None:
            expert_backbone = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "conf", "action_expert_backbone_small.json")
            )

        transformer_config = AutoConfig.from_pretrained(expert_backbone)
        self.VideoGPT = AutoModelForCausalLM.from_config(transformer_config, attn_implementation="eager")

        head_token_size = self.VideoGPT.config.hidden_size
        self.head_token_size = head_token_size

        self.action_head = nn.Sequential(  nn.Linear(self.head_token_size * (future_action_window_size + 1), self.head_token_size*4, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(self.head_token_size*4, self.head_token_size*2, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(self.head_token_size*2, self.head_token_size//2, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(self.head_token_size//2, (future_action_window_size + 1) * action_dim, bias=True),
                                        )
        vision_model = timm.create_model(  'vit_large_patch14_reg4_dinov2.lvd142m',
                                                pretrained=True,
                                                num_classes=0,  # remove classifier nn.Linear
                                                img_size = default_image_size
                                            )

        self.film_vision_model = FiLMedDinoVisionBackbone(vision_model, token_size)

        self.visual_projector = nn.Sequential(  nn.Linear(1024, self.head_token_size, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(self.head_token_size, self.head_token_size, bias=True),
                                                )

        self.cog_projector = nn.Sequential( nn.Linear(token_size, self.head_token_size, bias=True),
                                            nn.SiLU(),
                                            nn.Linear(self.head_token_size, self.head_token_size, bias=True),
                                            )
        
        self.default_image_size = default_image_size
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.film_vision_model)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        self.default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        
        self.future_action_window_size = future_action_window_size
        self.action_dim = action_dim
        
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
            self.head_token_size, 100.0
        )
        self.action_embed = ActionEncoder( head_token_size=self.head_token_size,
                                            action_dim=action_dim
                                        )
    
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
                latent_action: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                actions: Optional[torch.FloatTensor] = None,
                t: Optional[torch.FloatTensor] = None,):

        pixel_values = pixel_values['dino']
        
        batch_size = latent_action.shape[0]
        num_images = pixel_values.shape[0]

        repeated_latent_action = latent_action.repeat_interleave(num_images//batch_size, dim=0)
        visual_embed = self.film_vision_model(pixel_values, repeated_latent_action)

        visual_embed = visual_embed.view(batch_size, num_images//batch_size, 256, 1024) # [bs num_image image_tokens dim]
        visual_embed = visual_embed.permute(0, 2, 1, 3)
        visual_embed = visual_embed.reshape(batch_size, 256 * num_images//batch_size, 1024)

        visual_feature = self.visual_projector(visual_embed)
        latent_action = self.cog_projector(latent_action)

        # prepare flow matching variables
        x0 = torch.randn_like(actions, device=actions.device, dtype=actions.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        time_cond = self.time_embedding(t)
        action_embeds = self.action_embed(psi_t, time_cond)

        # prepare action queries
        input_seq = torch.cat([visual_feature, latent_action, action_embeds], dim=1)

        vis_len = visual_feature.shape[1]
        cog_len = latent_action.shape[1]
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
                latent_action: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                num_inference_steps: int = 10,
                ):
        # prepare features
        pixel_values = pixel_values['dino']
        device = latent_action.device
        dtype = latent_action.dtype

        # from IPython import embed;embed()
        batch_size = latent_action.shape[0]
        num_images = pixel_values.shape[0]

        repeated_latent_action = latent_action.repeat_interleave(num_images//batch_size, dim=0)
        visual_embed = self.film_vision_model(pixel_values, repeated_latent_action)

        visual_embed = visual_embed.view(batch_size, num_images//batch_size, 256, 1024) # [bs num_image image_tokens dim]
        visual_embed = visual_embed.permute(0, 2, 1, 3)
        visual_embed = visual_embed.reshape(batch_size, 256 * num_images//batch_size, 1024)

        visual_feature = self.visual_projector(visual_embed)
        latent_action = self.cog_projector(latent_action)

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

            input_seq = torch.cat([visual_feature, latent_action, action_embeds], dim=1)

            if attention_mask is None:
                vis_len = visual_feature.shape[1]
                cog_len = latent_action.shape[1]
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

class ActionModelWithState(ActionModel):
    def __init__(self, 
                 token_size,
                 past_action_window_size,
                 future_action_window_size,
                 action_dim,
                 expert_backbone=None,
                 default_image_size=224,
                 ):
        super().__init__(                 
                 token_size,
                 past_action_window_size,
                 future_action_window_size,
                 action_dim,
                 expert_backbone,
                 default_image_size)

        self.state_embed = nn.Sequential( nn.Linear(8, self.head_token_size, bias=True),
                                    nn.SiLU(),
                                    nn.Linear(self.head_token_size, self.head_token_size, bias=True),
                                    )
        
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
                latent_action: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                actions: Optional[torch.FloatTensor] = None,
                t: Optional[torch.FloatTensor] = None,
                proprios: Optional[torch.FloatTensor] = None,
                ):
        pixel_values = pixel_values['dino']

        batch_size = latent_action.shape[0]
        num_images = pixel_values.shape[0]

        # repeat latent action interleavingly for FiLM
        repeated_latent_action = latent_action.repeat_interleave(num_images//batch_size, dim=0)
        
        visual_embed = self.film_vision_model(pixel_values, repeated_latent_action)

        visual_embed = visual_embed.view(batch_size, num_images//batch_size, 256, 1024) # [bs num_image image_tokens dim]
        visual_embed = visual_embed.permute(0, 2, 1, 3)
        visual_embed = visual_embed.reshape(batch_size, 256 * num_images//batch_size, 1024)

        visual_feature = self.visual_projector(visual_embed)
        latent_action = self.cog_projector(latent_action)

        x0 = torch.randn_like(actions, device=actions.device, dtype=actions.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        time_cond = self.time_embedding(t)
        action_embeds = self.action_embed(psi_t, time_cond)

        proprios_embed = self.state_embed(proprios).unsqueeze(1)

        # prepare action queries
        input_seq = torch.cat([visual_feature, latent_action, proprios_embed, action_embeds], dim=1)

        vis_len = visual_feature.shape[1]
        cog_len = latent_action.shape[1] + 1 # include proprios_embed
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
                latent_action: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                num_inference_steps: int = 10,
                proprios: Optional[torch.FloatTensor] = None,
                ):
        # prepare features
        pixel_values = pixel_values['dino']
        device = latent_action.device
        dtype = latent_action.dtype

        # from IPython import embed;embed()
        batch_size = latent_action.shape[0]
        num_images = pixel_values.shape[0]

        repeated_latent_action = latent_action.repeat_interleave(num_images//batch_size, dim=0)
        visual_embed = self.film_vision_model(pixel_values, repeated_latent_action)

        visual_embed = visual_embed.view(batch_size, num_images//batch_size, 256, 1024) # [bs num_image image_tokens dim]
        visual_embed = visual_embed.permute(0, 2, 1, 3)
        visual_embed = visual_embed.reshape(batch_size, 256 * num_images//batch_size, 1024)

        visual_feature = self.visual_projector(visual_embed)
        latent_action = self.cog_projector(latent_action)

        proprios_embed = self.state_embed(proprios).unsqueeze(1)

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

            input_seq = torch.cat([visual_feature, latent_action, proprios_embed, action_embeds], dim=1)

            if attention_mask is None:
                vis_len = visual_feature.shape[1]
                cog_len = latent_action.shape[1] + 1 # include proprios_embed
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

