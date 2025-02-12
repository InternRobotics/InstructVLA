import deepspeed
from deepspeed import DeepSpeedConfig
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics, validation_VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer
from .base_strategy import TrainingStrategy
from collections import defaultdict
from deepspeed.accelerator import get_accelerator

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class ZerosStrategy(TrainingStrategy):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        deepspeed_config: Optional[dict] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size
        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm

        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.mixed_precision_dtype = mixed_precision_dtype
        self.worker_init_fn = worker_init_fn

        self.reduce_in_full_precision = reduce_in_full_precision
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        self.flow_sampling = "beta"
        flow_alpha = 1.5
        flow_beta = 1
        self.flow_t_max = 1 - 0.001
        self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
        self.multi_sampling = 2 # it will sample different time steps to the same batch working as grad acc
        # Initialize DeepSpeed
        self.deepspeed_config = deepspeed_config or {
            "train_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": 1,  # force to 1

            "zero_optimization": {
                "stage": 3,  # ZeRO Stage 3
                "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
                },
                "offload_param": {
                "device": "cpu",
                "pin_memory": True
                },
                "overlap_comm": True,  # Enable communication-computation overlap
                "contiguous_gradients": True,  # Optimized memory management
                "reduce_bucket_size": "auto",  # Auto-tune reduce bucket size
                "sub_group_size": 1e9,  # Ensure sub_group_size is an integer
                "allgather_partitions": True,  # Efficient parameter gathering
                "allgather_bucket_size": "auto",  # Increase bucket size for efficiency
                "reduce_scatter": True,  # Reduce communication overhead
                "stage3_prefetch_bucket_size": "auto",  # Optimize bucket size for prefetching
                "stage3_param_persistence_threshold": "auto",  # Improve parameter memory efficiency
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
            },
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "gradient_clipping": self.max_grad_norm,  # Keep existing gradient clipping
        }


        self.optimizer, self.lr_scheduler = None, None
    
    def run_setup(self, run_dir: Path, n_train_examples: int, **_: str,) -> None:
        """Set up optimizer, scheduler, and initialize DeepSpeed."""
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            lr_scheduler = {
                'scheduler': {
                    "type": "WarmupDecayLR",
                    "params": {
                        "warmup_min_lr": 5e-8,
                        "warmup_max_lr": self.learning_rate,
                        "warmup_num_steps": num_warmup_steps,
                        "total_num_steps": num_training_steps,
                    },
                }
            }
        elif self.lr_scheduler_type == "constant":
            lr_scheduler = {
                'scheduler': {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 5e-8,
                        "warmup_max_lr": self.learning_rate,
                        "warmup_num_steps": 200,
                    },
                }
            }
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {self.lr_scheduler_type}")

        self.deepspeed_config.update(lr_scheduler)
        

        # Create parameter groups
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.vlm.named_parameters() if p.requires_grad
                ],
                "name": "trainable_param",
            },
        ]

        optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)

        # Initialize DeepSpeed
        
        # Initialize DeepSpeed with DeepSpeedCPUAdam
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.vlm,
            config=self.deepspeed_config,
            optimizer=DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            ),
            model_parameters=optimizer_grouped_parameters,
        )


        self.vlm = model_engine
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    @staticmethod
    def preprocess_batch(batch, device, model_dtype):
        processed_batch = {
            "input_ids": batch["input_ids"].to(device=device),
            "attention_mask": batch["attention_mask"].to(device=device),
            "labels": batch["labels"].to(device=device),
            "action": batch["action"].to(device=device),
        }

        if isinstance(batch["pixel_values"], dict):
            processed_batch["pixel_values"] = {
                k: v.to(device=device, dtype=model_dtype) for k, v in batch["pixel_values"].items()
            }
        else:
            processed_batch["pixel_values"] = batch["pixel_values"].to(device=device, dtype=model_dtype)

        remaining_keys = set(batch.keys()) - set(processed_batch.keys()) - {"pixel_values"}
        processed_batch.update({k: batch[k] for k in remaining_keys})

        return processed_batch


    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        validation_set: IterableDataset = None,
        val_frequency: int = 5,
        validation_metrics: validation_VLAMetrics = None,
        val_steps: int = 1000
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        grad_accum_steps = self.multi_sampling #self.vlm.gradient_accumulation_steps()

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )
        if validation_set is not None:
            valid_dataloader = DataLoader(
            validation_set,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()
            device = self.vlm.device
            model_dtype = next(self.vlm.parameters()).dtype

            for batch in dataloader:

                # Mixed precision and forward pass
                batch = self.preprocess_batch(batch,device,model_dtype)
                loss_accumulated = 0

                for grad_acc_step in range(grad_accum_steps): # multi_sampling
                    with torch.autocast("cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training):
                        sampled_t = self.sample_fm_time(batch['input_ids'].shape[0]).to(dtype=model_dtype, device=device)

                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            pixel_values=batch['pixel_values'],
                            labels=batch['labels'],
                            action = batch['action'],
                            t=sampled_t,
                        )
                        loss = output.loss / grad_accum_steps

                    self.vlm.backward(loss)
                    loss_accumulated += loss

                grad_norm = compute_grad_norm(self.vlm)
                dist.all_reduce(grad_norm, op=dist.ReduceOp.SUM)
                metrics.commit(grad_norm=grad_norm)
                metrics.commit(loss=loss_accumulated)
                
                self.commit_metrics(metrics=metrics,
                    output=output,
                    metric_keys = { "moe_loss": "moe_loss",
                                    "action_loss": "action_loss",
                                    "expert_action_accuracy_01": "expert_action_accuracy_01",
                                    "expert_action_accuracy_05": "expert_action_accuracy_05",
                                    "expert_action_l1_loss": "expert_action_l1_loss",
                                    "mse_loss": "mse_loss"
                                    }
                    )

                # Update weights and LR scheduler using DeepSpeed
                self.vlm.step()
                get_accelerator().empty_cache()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )
                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)
                
                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()
                if validation_set is not None and (metrics.global_step % (save_interval//val_frequency)) == 0:
                    val_metrics = defaultdict(list)
                    with torch.no_grad():
                        self.vlm.eval()
                        val_count = 0
                        for batch in tqdm(valid_dataloader,disable=not overwatch.is_rank_zero(),):
                            if val_count>=val_steps:
                                break
                            val_count +=1
                            with torch.autocast(
                                "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                                ):
                                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                                    # Mixed precision and forward pass

                                    batch = self.preprocess_batch(batch,device,model_dtype)
                                    output: CausalLMOutputWithPast = self.vlm(
                                        input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        pixel_values=batch['pixel_values'],
                                        labels=batch['labels'],
                                        action = batch['action'],
                                        inference_pass=True,
                                    )

                                    val_metrics['l1_loss'].append(output['l1_loss'].item())
                                    val_metrics['mse_loss'].append(output['mse_loss'].item())
                                    val_metrics['accuracy05'].append(output['accuracy05'].item())
                                    val_metrics['accuracy005'].append(output['accuracy005'].item())
                                    val_metrics['accuracy01'].append(output['accuracy01'].item())

                        mean_metrics = {}
                        for key, values in val_metrics.items():
                            local_sum = torch.tensor(sum(values), device=action_preds.device, dtype=torch.float32)
                            local_count = torch.tensor(len(values), device=action_preds.device, dtype=torch.float32)

                            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

                            global_mean = local_sum / local_count
                            mean_metrics[key] = global_mean

                        mean_metrics['global_step'] = metrics.global_step
                        # commit & push only at rank zero
                        validation_metrics.commit(**mean_metrics)
                        validation_metrics.push()
                        self.vlm.train()

                # Save checkpoint periodically
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    # deepspeed will handle the multi-thread sync
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update progress bar
                progress.update()
                progress.set_description(status)
    
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
    ) -> None:
        """Save a checkpoint to the `run_dir` using DeepSpeed."""
        assert isinstance(self.vlm, deepspeed.DeepSpeedEngine), "save_checkpoint assumes VLM is wrapped in DeepSpeed!"

        # Set Checkpoint Path
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tag = f"step-{global_step:06d}-epoch-{epoch:02d}"

        # Gather full model parameters
        full_state_dict = {}
        for name, param in self.vlm.module.named_parameters():
            with deepspeed.zero.GatheredParameters([param]):
                full_state_dict[name] = param.data.detach().cpu().clone()

        # Save the full state dict
        if overwatch.is_rank_zero():
            checkpoint_path = checkpoint_dir / f"{tag}-model.pth"
            torch.save({"epoch": epoch, "global_step": global_step, "model": full_state_dict}, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    def clip_grad_norm(self) -> None:
        pass


def compute_grad_norm(model):
    """Compute gradient norm, considering DeepSpeed model partitioning."""
    if hasattr(model, "module"):
        model = model.module  

    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]

    if hasattr(model, "get_model_norm"):
        total_norm = model.get_model_norm(2)
    else:
        for p in parameters:
            param_norm = p.grad.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

    return torch.tensor(total_norm, device=next(model.parameters()).device)

