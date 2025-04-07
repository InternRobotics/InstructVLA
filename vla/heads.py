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
                 head_token_size = 768
                 ):
        super().__init__()
        self.action_head = nn.Sequential(  nn.Linear(head_token_size * (future_action_window_size + 1), head_token_size*4, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(head_token_size*4, head_token_size*2, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(head_token_size*2, head_token_size//2, bias=True),
                                                nn.Linear(head_token_size//2, (future_action_window_size + 1) * action_dim, bias=True),
                                        ) # 120M
        self.vision_model = timm.create_model(  'vit_large_patch14_reg4_dinov2.lvd142m',
                                                pretrained=True,
                                                num_classes=0,  # remove classifier nn.Linear
                                                img_size = 224
                                            )

        self.vision_model.load_state_dict(torch.load(pertrained_dino))

        self.time_embedding = SinusoidalPosEmb(
                head_token_size, 10000.0
            )


        self.vision_model.forward = unpack_tuple(
            partial(self.vision_model.get_intermediate_layers, n={len(self.vision_model.blocks) - 2})
        )

        self.visual_projector = nn.Sequential(  nn.Linear(1024, head_token_size, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(head_token_size, head_token_size, bias=True),
                                                )
        
        self.cog_projector = nn.Sequential(     nn.Linear(token_size, head_token_size, bias=True),
                                                nn.SiLU(),
                                                nn.Linear(head_token_size, head_token_size, bias=True),
                                                )

        self.action_embed = ActionEncoder( head_token_size=head_token_size,
                                            action_dim=action_dim
                                        )
        self.future_action_window_size = future_action_window_size
        self.action_dim = action_dim
        
        if VQTokenizer is None or VideoGPT is None:
            print("load world model from local path")
            self.VQTokenizer = CompressiveVQModel.from_pretrained(world_model, subfolder='tokenizer', low_cpu_mem_usage=False)
            self.VideoGPT = AutoModelForCausalLM.from_pretrained(world_model, subfolder='transformer', low_cpu_mem_usage=False)
        else:
            self.VQTokenizer = VQTokenizer
            self.VideoGPT = VideoGPT

        self.flow_sig_min = 0.001
        

        # self.VideoGPT.gradient_checkpointing_enable()
            
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
                indices_for_now = None,
                actions: Optional[torch.FloatTensor] = None,
                t: Optional[torch.FloatTensor] = None,
                ):

        batch_size = cognition_features.shape[0]
        # BS, 256,
        visual_embed = self.vision_model.forward(pixel_values['dino'][indices_for_now])

        visual_feature = torch.zeros_like(self.visual_projector(visual_embed), device=actions.device, dtype=actions.dtype)
        cognition_features = self.cog_projector(cognition_features)

        x0 = torch.randn_like(actions, device=actions.device, dtype=actions.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        time_cond = self.time_embedding(t)
        action_embeds = self.action_embed(psi_t, time_cond)

        input_seq = torch.cat([visual_feature, cognition_features, action_embeds], dim=1)

        vis_len = visual_feature.shape[1]
        cog_len = cognition_features.shape[1]
        fut_len = action_embeds.shape[1]

        
        total_len = vis_len + cog_len + fut_len
        # attention_mask = torch.ones((total_len, total_len), dtype=torch.bool, device=input_seq.device)
        # blockwise causal attention mask
        attention_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=input_seq.device)
        attention_mask[:vis_len, :vis_len] = 1
        attention_mask[vis_len:vis_len+cog_len, :vis_len+cog_len] = 1
        attention_mask[vis_len+cog_len:, :total_len] = 1

        # make it 4d
        # [BS, 1, q_len, kv_len]
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, total_len, total_len)

        encoded_seq = self.VideoGPT(inputs_embeds=input_seq,
                                    attention_mask=attention_mask,
                                    use_cache=False,
                                    output_hidden_states=True,
                                    return_dict=True)
    
        future_pred = encoded_seq.hidden_states[-1][:,-(self.future_action_window_size + 1):]
        future_pred = future_pred.reshape(batch_size, -1)

        output = self.action_head(future_pred)
        v_psi  = output.reshape(batch_size, self.future_action_window_size + 1, self.action_dim)
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        # from IPython import embed;embed()
        return torch.mean((v_psi - d_psi) ** 2)
