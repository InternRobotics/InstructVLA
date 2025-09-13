"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import sys
sys.path.insert(0, os.getcwd())
import draccus
import torch
import torch.distributed as dist
import yaml
import wandb

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from training import Metrics, get_train_strategy
from conf import VLAConfig, VLARegistry
from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load, load_vla, get_vla_dataset_and_collator

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.QWEN25_DINOSIGLIP_224PX_0_5B_MAGIC_SOUP_PLUS_MINUS.vla_id)
    )

    # Directory Paths
    data_root_dir: str = str(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    debug: bool = False

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Union[str, Path]] = None                  # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    # (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 42                                                  # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = ".hf_token"                        # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    #trackers: Tuple[str, ...] = ("jsonl",)                         # Trackers to initialize (if W&B, add config!)
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    repeated_diffusion_steps: int = 8                               # Repeated steps for training action model (a diffusion model)
    load_all_data_for_training: bool = True                         # Load all training data 
    future_action_window_size: int = 15                             # Action chunking, predicting future actions + current action
    past_action_window_size: int = 0                                # Action history window size, not used now, set to be 0 

    action_dim: int = 7                                             # Dimension of action space
    use_mm: bool = True
    stage: str = "stage1"
    fix_system1: bool = False
    num_of_meta_query: int = 64
    disable_instruction: bool = False

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        overwatch.warning(f"Remove max steps for VLM-only training")
        self.max_steps = None # only for VLM training!!!
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        if self.debug:
            if (
                self.vla.expected_world_size == overwatch.world_size()
            ):
                overwatch.warning(f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!")
        else:
            assert self.vla.expected_world_size == overwatch.world_size(),f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("Instruct-VLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    cfg.run_id += cfg.stage

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = cfg.hf_token
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
    
    dist.barrier()
    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.debug:
        overwatch.warning(f"!!!!!!!!!!!!!!!!!!!! you are under the debugging mode, restrictions about GPU and load pretrained weights are removed !!!!!!!!!!!!!!!!!!!")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        # if cfg.is_resume:
        #     assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
        #     assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch
        overwatch.info(f"Loading VLA {cfg.stage} Checkpoint")
        
        vla = load_vla(cfg.pretrained_checkpoint, 
                        load_for_training=not cfg.debug, 
                        action_dim=cfg.action_dim,
                        future_action_window_size=cfg.future_action_window_size,
                        past_action_window_size=cfg.past_action_window_size,
                        stage=cfg.stage,
                        num_of_meta_query=cfg.num_of_meta_query
                        )

    else:
        vlm = load(cfg.vla.base_vlm, load_for_training=not cfg.debug, stage=cfg.stage, num_of_meta_query=cfg.num_of_meta_query)
        overwatch.info("Creating VLA from Base VLM")

        vla = vlm
        vla.past_action_window_size = cfg.past_action_window_size
        vla.future_action_window_size = cfg.future_action_window_size


    # [Validate] Model should be in Full Precision!
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    overwatch.warning(
        f"Removing the action head beacuse we only finetune the llm backbone!!!"
    )
    filtered_trainable_keys = []
    for key in vla._trainable_module_keys:
        if key != 'action_model':
            filtered_trainable_keys.append(key)
    vla._trainable_module_keys = filtered_trainable_keys
    del vla.action_model


    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")

    # from torch.utils.data import ConcatDataset
    # from mm_dataset.data_utils import LazyPointingDataset, LazyPointDetectionDataset, DataCollatorForSupervisedDataset

    # pointing_dataset = LazyPointingDataset(tokenizer=vla.tokenizer, processor=vla.processor)
    # detection_dataset = LazyPointDetectionDataset(tokenizer=vla.tokenizer, processor=vla.processor)

    # mm_dataset = ConcatDataset([pointing_dataset, detection_dataset])
    # mm_collator = DataCollatorForSupervisedDataset(tokenizer=vla.tokenizer)

    from mm_dataset.data_utils import LazyPointingDataset, LazyPointDetectionDataset, LazySupervisedDataset, DataCollatorForSupervisedDataset
    mm_dataset = LazySupervisedDataset(tokenizer=vla.tokenizer,
                                        processor=vla.processor,
                                        data_path='bunny_dataset/finetune/bunny_695k.json',
                                        )

    mm_collator=DataCollatorForSupervisedDataset(tokenizer=vla.tokenizer)

    # remove action dataset because we only finetunng language output
    # Save dataset statistics for de-normalization at inference time
    dist.barrier()
    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")

    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vla,
        device_id=device_id,
        stage="full-finetune",
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
        repeated_diffusion_steps=cfg.repeated_diffusion_steps,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(mm_dataset))
    if cfg.pretrained_checkpoint is not None and cfg.is_resume:
        train_strategy.load_optimizer_and_scheduler(cfg.pretrained_checkpoint)

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        stage='finetune',
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")

    train_strategy.run_training(
        metrics = metrics,
        dataset=mm_dataset,
        collator=mm_collator,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
