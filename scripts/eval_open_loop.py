from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
import rootutils
import torch

from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import tqdm

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split('/')[int(idx)])

# Add the project root directory to the Python path
# rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from galaxea_fm.data.galaxea_lerobot_dataset import GalaxeaLerobotDataset
from galaxea_fm.models.galaxea_zero.galaxea_zero_policy import GalaxeaZeroPolicy
from galaxea_fm.utils.pytorch_utils import dict_apply, dict_to_array, set_global_seed
from galaxea_fm.utils.visualize import plot_result
from galaxea_fm.utils.normalizer import load_dataset_stats_from_json
from galaxea_fm.processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path


def plot_open_ring(
    gt_action: npt.NDArray,
    inferred_action: npt.NDArray,
    output_dir: Path,
    prefix: str = "action_comparison",
    return_fig: bool = False,
):
    """
    Plot comparison of GT and inferred action sequences for all dimensions in a single figure.

    Args:
        gt_action: Ground truth action sequence, shape (T, D) where T is time steps, D is dimensions
        inferred_action: Inferred action sequence, shape (T, D)
        output_dir: Output directory path
        prefix: Prefix for the saved plot filename
    """
    # Convert to numpy arrays if needed
    if not isinstance(gt_action, np.ndarray):
        gt_action = np.array(gt_action)
    if not isinstance(inferred_action, np.ndarray):
        inferred_action = np.array(inferred_action)

    # Ensure actions are 2D
    if gt_action.ndim == 1:
        gt_action = gt_action.reshape(-1, 1)
    if inferred_action.ndim == 1:
        inferred_action = inferred_action.reshape(-1, 1)

    # Get dimensions
    T_gt, D_gt = gt_action.shape
    T_inf, D_inf = inferred_action.shape

    if D_gt != D_inf:
        print(
            f"Warning: GT has {D_gt} dimensions, Inferred has {D_inf} dimensions"
        )
        D = min(D_gt, D_inf)
    else:
        D = D_gt

    # Create a large figure with subplots for all dimensions
    cols = 3
    rows = (D + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    # Flatten axes array for easier iteration
    if D == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Time axes
    time_gt = np.arange(T_gt)
    time_inf = np.arange(T_inf)

    # Plot each dimension
    for dim in range(D):
        ax = axes[dim]

        # Plot both actions for this dimension
        ax.plot(
            time_gt,
            gt_action[:, dim],
            "-",
            label="GT",
            linewidth=2,
            markersize=4,
            color="tab:blue",
        )
        ax.plot(
            time_inf,
            inferred_action[:, dim],
            "--",
            label="Inferred",
            linewidth=2,
            markersize=4,
            color="tab:orange",
        )

        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_title(f"Dimension {dim}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for dim in range(D, len(axes)):
        axes[dim].set_visible(False)

    plt.suptitle("Action Comparison: GT vs Inferred",
                 fontsize=16,
                 fontweight="bold",
                 y=0.995)
    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{prefix}.png"
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    print(f"Saved action comparison plot with {D} dimensions to {save_path}")

    if return_fig:
        return fig
    plt.close()
    return None

def sample_trajectory(
    cfg: DictConfig,
    model: GalaxeaZeroPolicy,
    loader: torch.utils.data.DataLoader,
    processor: BaseProcessor,
    max_len: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    action_horizon = cfg.model.model_arch.horizon_steps
    max_iter = max_len // action_horizon
    traj_actions = []
    traj_gt_actions = []
    with tqdm.tqdm(total=max_iter, desc="Sampling trajectory") as pbar:
        for i, batch in enumerate(loader):
            if i % action_horizon != 0:
                continue
            if i // action_horizon >= max_iter:
                break
            batch = dict_apply(batch, lambda x: x.cuda() if isinstance(x, torch.Tensor) else x)
            with torch.no_grad():
                batch = model.predict_action(batch)

            batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
            batch = processor.postprocess(batch)
            cur_pd_action = dict_apply(batch["action"], lambda x: x.cpu().numpy())
            cur_gt_action = dict_apply(batch["gt_action"], lambda x: x.cpu().numpy())
            traj_actions.append(dict_to_array(cur_pd_action)[0])
            traj_gt_actions.append(dict_to_array(cur_gt_action)[0])
            pbar.update(1)
    traj_actions = np.concatenate(traj_actions, axis=0)
    traj_gt_actions = np.concatenate(traj_gt_actions, axis=0)
    return traj_actions, traj_gt_actions


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed"):
        set_global_seed(cfg.seed, get_worker_init_fn=False)

    # load model
    model: GalaxeaZeroPolicy = instantiate(cfg.model.model_arch)
    use_lora = cfg.model.get("use_lora", False)
    if use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA but not installed. Install it with: pip install peft")
        
        lora_config = cfg.model.get("lora", {})
        logger.info("Applying LoRA to model...")
        logger.info(f"LoRA config: {lora_config}")
        
        # Get the underlying model (GalaxeaZero) from the policy
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
        
        # Apply LoRA to different components based on config
        lora_target_modules = lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        lora_r = lora_config.get("r", 8)
        lora_alpha = lora_config.get("alpha", 16)
        lora_dropout = lora_config.get("dropout", 0.1)
        lora_bias = lora_config.get("bias", "none")
        
        # Determine which parts of the model to apply LoRA to
        apply_to_vlm = lora_config.get("apply_to_vlm", True)
        apply_to_action = lora_config.get("apply_to_action", False)
        
        if apply_to_vlm and hasattr(base_model, 'joint_model') and hasattr(base_model.joint_model, 'mixtures'):
            # Apply LoRA to VLM (language model) part
            if "vlm" in base_model.joint_model.mixtures:
                vlm_mixture = base_model.joint_model.mixtures["vlm"]
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    target_modules=lora_target_modules,
                )
                base_model.joint_model.mixtures["vlm"] = get_peft_model(vlm_mixture, peft_config)
                logger.info(f"Applied LoRA to VLM mixture with r={lora_r}, alpha={lora_alpha}")
        
        if apply_to_action and hasattr(base_model, 'joint_model') and hasattr(base_model.joint_model, 'mixtures'):
            # Apply LoRA to action expert part
            if "action" in base_model.joint_model.mixtures:
                action_mixture = base_model.joint_model.mixtures["action"]
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    target_modules=lora_target_modules,
                )
                base_model.joint_model.mixtures["action"] = get_peft_model(action_mixture, peft_config)
                logger.info(f"Applied LoRA to action mixture with r={lora_r}, alpha={lora_alpha}")
        

    state_dict = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    # HACK: ignore normalizer keys for testing using v1.0.0 checkpoints
    model.load_state_dict(state_dict, strict=False)
    policy = model.cuda().eval()
    logger.info(f"Model loaded")
    
    dataset_val: GalaxeaLerobotDataset = instantiate(cfg.data, is_training_set=False)

    dataloader = DataLoader(
        dataset_val, 
        shuffle=False, 
        batch_size=cfg.batch_size_val, 
        num_workers=cfg.model.num_workers, 
        pin_memory=cfg.model.pin_memory, 
        persistent_workers=cfg.model.persistent_workers, 
        worker_init_fn=None, 
    )
    # NOTE: use pretrained norm stats
    checkpoint_path = Path(cfg.ckpt_path)
    dataset_stats = load_dataset_stats_from_json(checkpoint_path.parent.parent / "dataset_stats.json")
    processor: BaseProcessor = instantiate(cfg.model.processor)

    processor.set_normalizer_from_stats(dataset_stats)
    dataset_val.set_processor(processor)
    

    actions, gt_actions = sample_trajectory(cfg, model, dataloader, processor, max_len=cfg.eval_steps)
    plot_open_ring(
        gt_actions,
        actions,
        output_dir=Path(cfg.ckpt_path).parent / "open_loop",
    )


if __name__ == "__main__":
    main()
