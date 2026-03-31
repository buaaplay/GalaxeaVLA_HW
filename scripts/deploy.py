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

# For Deploy
import requests
import base64
import time
import cv2


def decode_image(b64_string):
    """Decode a base64-encoded string into an OpenCV image."""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def save_image(b64_string, file_name):
    """Save a base64-encoded image to the specified file path."""
    if not b64_string:
        print(f"[{file_name}] No image data received.")
        return

    img = decode_image(b64_string)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    cv2.imwrite(file_name, img)
    print(f"Image saved to: {file_name}")


# batch["input_ids"], (1, 823)
# batch["attention_mask"], (1, 823)
# batch["pixel_values"], (1, 3, 3, 224, 224)
# batch["proprio"], (1, 1, 14)
#
# ['input_ids', 'labels', 'attention_mask', 'pixel_values', 'gt_action', 'action', 'action_is_pad', 'action_dim_is_pad', 'proprio', 'proprio_is_pad', 'proprio_dim_is_pad', 'idx']


def make_data(inputs):
    # Processor input:
    #   Data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
    #             - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
    #             - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
    #             - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
    #             - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
    #             - "state_is_pad": torch.Tensor -> [num_obs_steps,]
    #             - "image_is_pad": torch.Tensor -> [num_obs_steps,]
    #             - "idx": int, sample index
    sample = {
        "task": inputs['prompt'],
        # "action": ,
        "state": {
            'left_arm': torch.tensor(inputs['states']['left_arm']).unsqueeze(0).to(torch.float32),
            'left_gripper': torch.tensor(inputs['states']['left_gripper']).unsqueeze(0).to(torch.float32),
            'right_arm': torch.tensor(inputs['states']['right_arm']).unsqueeze(0).to(torch.float32),
            'right_gripper': torch.tensor(inputs['states']['right_gripper']).unsqueeze(0).to(torch.float32),
        },
        "images": {
            'head_rgb': torch.tensor(inputs['images']['head']).permute(2, 0, 1).unsqueeze(0),  # 1280*720
            'left_wrist_rgb': torch.tensor(inputs['images']['left_wrist']).permute(2, 0, 1).unsqueeze(0),  # 640*360
            'right_wrist_rgb': torch.tensor(inputs['images']['right_wrist']).permute(2, 0, 1).unsqueeze(0),  # 640*360
        },
        # "action_is_pad": torch.zeros(1),
        "state_is_pad": torch.zeros(1),
        "image_is_pad": torch.zeros(1),
        "idx": 0,
    }

    return sample


def infer(model, processor, inputs):
    sample = make_data(inputs)
    sample = processor.preprocess(sample)

    # import ipdb;ipdb.set_trace()
    batch = dict_apply(sample, lambda x: x.unsqueeze(0).cuda() if isinstance(x, torch.Tensor) else x)
    with torch.no_grad():
        batch = model.predict_action(batch)

    batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)

    # print(batch['action'][0][:, 7:13])
    batch = processor.postprocess(batch)
    # print('*'*50)
    # print(batch['action']['right_arm'][0])
    # import ipdb;ipdb.set_trace()

    res = {}
    for key, value in batch['action'].items():
        res[key] = value.squeeze(0).numpy()
    return res


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="train.yaml")
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
            raise ImportError(
                "peft is required for LoRA but not installed. Install it with: pip install peft"
            )

        lora_config = cfg.model.get("lora", {})
        logger.info("Applying LoRA to model...")
        logger.info(f"LoRA config: {lora_config}")

        # Get the underlying model (GalaxeaZero) from the policy
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model

        # Apply LoRA to different components based on config
        lora_target_modules = lora_config.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        lora_r = lora_config.get("r", 8)
        lora_alpha = lora_config.get("alpha", 16)
        lora_dropout = lora_config.get("dropout", 0.1)
        lora_bias = lora_config.get("bias", "none")

        # Determine which parts of the model to apply LoRA to
        apply_to_vlm = lora_config.get("apply_to_vlm", True)
        apply_to_action = lora_config.get("apply_to_action", False)

        if apply_to_vlm and hasattr(base_model, 'joint_model') and hasattr(
                base_model.joint_model, 'mixtures'):
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
                base_model.joint_model.mixtures["vlm"] = get_peft_model(
                    vlm_mixture, peft_config)
                logger.info(
                    f"Applied LoRA to VLM mixture with r={lora_r}, alpha={lora_alpha}"
                )

        if apply_to_action and hasattr(base_model, 'joint_model') and hasattr(
                base_model.joint_model, 'mixtures'):
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
                base_model.joint_model.mixtures["action"] = get_peft_model(
                    action_mixture, peft_config)
                logger.info(
                    f"Applied LoRA to action mixture with r={lora_r}, alpha={lora_alpha}"
                )

    state_dict = torch.load(cfg.ckpt_path,
                            map_location="cpu",
                            weights_only=False)["model_state_dict"]
    # HACK: ignore normalizer keys for testing using v1.0.0 checkpoints
    model.load_state_dict(state_dict, strict=False)
    policy = model.cuda().eval()
    logger.info(f"Model loaded")

    # NOTE: use pretrained norm stats
    checkpoint_path = Path(cfg.ckpt_path)
    dataset_stats = load_dataset_stats_from_json(
        checkpoint_path.parent.parent / "dataset_stats.json")
    processor: BaseProcessor = instantiate(cfg.model.processor)

    processor.set_normalizer_from_stats(dataset_stats)
    processor.eval()

    session = requests.Session()
    session.trust_env = False

    print(
        f"Starting cyclic communication, target address: {cfg.deploy.host}:{cfg.deploy.port}, loop interval: {cfg.deploy.loop_interval}s"
    )
    print("Press Ctrl+C to stop the loop")

    try:
        while True:
            t0 = time.time()
            data = None

            get_state_url = f"http://{cfg.deploy.host}:{cfg.deploy.port}/get_state"
            try:
                response = session.get(get_state_url, timeout=2)
                latency = (time.time() - t0) * 1000

                if response.status_code == 200:
                    data = response.json()
                    print(
                        f"\n=== Cyclic communication successful | Latency: {latency:.2f}ms ==="
                    )

                    print(f"Left Arm Angles:  {data['qpos_arm_left']}")
                    print(f"Left Gripper:     {data['qpos_gripper_left']}")

                    if data.get('image_head_base64'):
                        print("Head image received, saving...")
                        # Use the deploy directory from args to construct the file path
                        head_image_path = os.path.join(cfg.deploy.deploy_dir,
                                                       "received_head.jpg")
                        save_image(data['image_head_base64'], head_image_path)
                    else:
                        print("No head image in payload.")

                    if data.get('image_wrist_left_base64'):
                        wrist_left_image_path = os.path.join(
                            cfg.deploy.deploy_dir, "received_wrist_left.jpg")
                        save_image(data['image_wrist_left_base64'],
                                   wrist_left_image_path)

                    if data.get('image_wrist_right_base64'):
                        wrist_right_image_path = os.path.join(
                            cfg.deploy.deploy_dir, "received_wrist_right.jpg")
                        save_image(data['image_wrist_right_base64'],
                                   wrist_right_image_path)
                else:
                    print(
                        f"\nFailed to get state, response code: {response.status_code}"
                    )

            except requests.exceptions.RequestException as e:
                print(f"\nException occurred while fetching state: {e}")
                time.sleep(cfg.deploy.loop_interval)
                continue

            if data is None:
                time.sleep(cfg.deploy.loop_interval)
                continue

            try:
                head_image = decode_image(data["image_head_base64"])
                left_wrist_image = decode_image(
                    data["image_wrist_left_base64"])
                right_wrist_image = decode_image(
                    data["image_wrist_right_base64"])

                # Use the default prompt from args instead of hardcoded "hello"
                pi0_inputs = {
                    "images": {
                        "head": head_image,
                        "left_wrist": left_wrist_image,
                        "right_wrist": right_wrist_image,
                    },
                    "states": {
                        "left_arm": np.array(data["qpos_arm_left"]),
                        "right_arm": np.array(data["qpos_arm_right"]),
                        "left_eef": np.zeros(6),
                        "right_eef": np.zeros(6),
                        "left_gripper": np.array(data["qpos_gripper_left"]),
                        "right_gripper": np.array(data["qpos_gripper_right"]),
                    },
                    "prompt": cfg.deploy.prompt if cfg.deploy.prompt else "",
                }
            except Exception as e:
                print(f"Policy inference exception: {e}")
                time.sleep(cfg.deploy.loop_interval)
                continue

            actions_map = infer(policy, processor, pi0_inputs)
            base_action = [
                data['qpos_arm_left'] + data['qpos_arm_right'] +
                data['qpos_torso'][:3] + [0.0] + [0.0] + [0.0, 0.0, 0.0]
            ]
            actions = np.array(base_action * 32)
            actions[:, 0:6] = actions_map["left_arm"] #* 0
            actions[:, 15:16] = actions_map["left_gripper"] #* 10
            actions[:, 6:12] = actions_map["right_arm"] #* 0
            actions[:, 16:17] = actions_map["right_gripper"] #* 10

            binary_map = lambda x: np.where(x < 70, 0, 1) * x
            actions[:, 15] = binary_map(actions[:, 15])
            actions[:, 16] = binary_map(actions[:, 16])

            actions = actions[:cfg.deploy.use_chunk, :]
            actions = actions.tolist()

            try:
                apply_action_url = f"http://{cfg.deploy.host}:{cfg.deploy.port}/apply_action"
                payload = {"action_chunk": actions}
                response = requests.post(apply_action_url, json=payload)

                if response.status_code == 200:
                    print(
                        f"Action sent successfully | Response: {response.json()}"
                    )
                else:
                    print(
                        f"Failed to send action | Response code: {response.status_code}"
                    )

            except Exception as e:
                print(f"Exception occurred while sending action: {e}")

            time.sleep(cfg.deploy.loop_interval)

    except KeyboardInterrupt:
        print("\n\nCyclic communication stopped manually by the user")
        if data is not None:
            base_action = [
                [0.0] * 12 +
                data['qpos_torso'][:3] + [100.0] + [100.0] + [0.0, 0.0, 0.0]
            ]
            actions = np.array(base_action * 50)
            actions = actions.tolist()
            apply_action_url = f"http://{cfg.deploy.host}:{cfg.deploy.port}/apply_action"
            payload = {"action_chunk": actions}
            response = requests.post(apply_action_url, json=payload)
    finally:
        # Close the requests session to release resources
        session.close()
        print("Session closed, program exited normally")


if __name__ == "__main__":
    main()
