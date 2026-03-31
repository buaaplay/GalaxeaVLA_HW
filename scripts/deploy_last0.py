"""
deploy_last0.py

Deploy a LaST0 finetuned checkpoint on Galaxea R1-Lite.

Architecture:
  Real robot (HTTP server)          This script (inference client)
  ┌──────────────────┐              ┌──────────────────────────────────┐
  │  GET /get_state  │ ──obs──────▶ │  pack_inputs()                   │
  │  POST /apply_action│◀──action── │  get_action() [LaST0 runtime]    │
  └──────────────────┘              │  interpret_action() [delta→pose] │
                                    └──────────────────────────────────┘

State space  (16D):  [left_pos3, left_rotvec3, left_gripper2,
                      right_pos3, right_rotvec3, right_gripper2]
Action space (14D):  [left_dpos3, left_drotvec3, left_gripper1,
                      right_dpos3, right_drotvec3, right_gripper1]

The robot HTTP server is expected to return (in /get_state JSON):
  - image_head_base64
  - image_wrist_left_base64
  - image_wrist_right_base64
  - left_ee_pose   : list[7]  (pos3 + quat4, xyzw)
  - right_ee_pose  : list[7]
  - left_gripper   : float    (normalised 0~1)
  - right_gripper  : float
  - qpos_torso     : list[N]  (used to fill the wide action vector)

/apply_action expects {"action_chunk": [[...wide_dim...], ...]}
Wide action vector layout (same as existing deploy.py):
  [0:6]   left arm joints  (filled with EE_POSE target converted to joints,
                             OR left as current if EE_POSE mode is used)
  [6:12]  right arm joints
  [12:15] torso (3D, kept at current)
  [15]    left gripper
  [16]    right gripper
  [17:20] zeros

NOTE: Because the robot controller defaults to EE_POSE mode, we send the
      integrated target EEF pose directly in the wide vector positions that
      the controller reads for EE_POSE.  The exact slot mapping must be
      confirmed against the live EFMNode config; the current implementation
      uses the same slot layout as deploy.py (joint-angle slots) as a safe
      fallback and adds a clearly-marked TODO for EE_POSE slot remapping.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import hydra
import numpy as np
import requests
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation as R

# ── LaST0 imports (repo must be on PYTHONPATH) ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))           # GalaxeaVLA_HW root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "last0"))  # last0 inside GalaxeaVLA_HW
# janus is a subpackage of last0, do NOT add last0/janus to sys.path

from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor, ActionTokenizer
from experiments.robot.robot_utils import get_action

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ── helpers ──────────────────────────────────────────────────────────────────

def decode_image_to_pil(b64_string: str) -> Image.Image:
    """Decode base64 JPEG/PNG → PIL RGB image."""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def quat_xyzw_to_rotvec(quat_xyzw: List[float]) -> np.ndarray:
    """Convert quaternion (xyzw) to rotation vector (axis-angle, 3D)."""
    x, y, z, w = quat_xyzw
    return R.from_quat([x, y, z, w]).as_rotvec()


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to quaternion (xyzw)."""
    return R.from_rotvec(rotvec).as_quat()  # returns xyzw


def ee_pose_to_state_vec(ee_pose_7: List[float], gripper: float) -> np.ndarray:
    """
    Convert a single arm's EEF pose + gripper to the 8D state vector used
    during LaST0 training:
        [pos3, rotvec3, gripper, gripper]   (gripper duplicated, see docs)
    """
    pos = np.array(ee_pose_7[:3], dtype=np.float32)
    rotvec = quat_xyzw_to_rotvec(ee_pose_7[3:7]).astype(np.float32)
    g = float(gripper)
    return np.concatenate([pos, rotvec, [g, g]])


def build_state_16d(data: Dict[str, Any]) -> np.ndarray:
    """
    Build the 16D state vector from /get_state response.
    Layout: [left_pos3, left_rotvec3, left_gripper2,
             right_pos3, right_rotvec3, right_gripper2]
    """
    left = ee_pose_to_state_vec(data["left_ee_pose"], data["left_gripper"])
    right = ee_pose_to_state_vec(data["right_ee_pose"], data["right_gripper"])
    return np.concatenate([left, right]).astype(np.float32)


def integrate_delta_action(
    current_ee_pose_7: List[float],
    dpos: np.ndarray,
    drotvec: np.ndarray,
) -> np.ndarray:
    """
    Integrate a delta EEF action onto the current pose.
    Returns new pose as [pos3, quat4_xyzw] (7D).
    """
    pos = np.array(current_ee_pose_7[:3]) + dpos
    cur_rot = R.from_quat(current_ee_pose_7[3:7])   # xyzw
    delta_rot = R.from_rotvec(drotvec)
    new_rot = (delta_rot * cur_rot).as_quat()         # xyzw
    return np.concatenate([pos, new_rot]).astype(np.float32)


def load_stats(stats_path: str) -> Dict[str, Any]:
    """
    Load stats_data.json produced by LaST0 train.py.
    Returns a flat dict with keys: action_mask, action_min, action_max,
                                    state_mask, state_min, state_max
    """
    with open(stats_path, "r") as f:
        raw = json.load(f)
    # top-level key is the dataset name (e.g. "20260127")
    dataset_key = next(iter(raw))
    d = raw[dataset_key]
    return {
        "action_mask": np.array(d["action"]["mask"]),
        "action_min":  np.array(d["action"]["q01"]),
        "action_max":  np.array(d["action"]["q99"]),
        "state_mask":  np.array(d["state"]["mask"]),
        "state_min":   np.array(d["state"]["q01"]),
        "state_max":   np.array(d["state"]["q99"]),
    }


# ── action interpretation ─────────────────────────────────────────────────────

def build_wide_action_chunk(
    action_chunk: np.ndarray,          # [use_chunk, 14]
    data: Dict[str, Any],
    use_chunk: int,
) -> List[List[float]]:
    """
    Convert LaST0 14D delta-EEF action chunk to the wide action vector
    expected by /apply_action.

    Wide vector layout (20D, same as existing deploy.py):
      [0:6]   left arm  (here: integrated EEF pos3 + rotvec3 as placeholder)
      [6:12]  right arm (here: integrated EEF pos3 + rotvec3 as placeholder)
      [12:15] torso (kept at current)
      [15]    left gripper
      [16]    right gripper
      [17:20] zeros

    TODO: If the robot controller reads EE_POSE from a different slot range,
          update the index assignments below to match EFMNode's action vector
          layout for EE_POSE mode.
    """
    torso = data.get("qpos_torso", [0.0, 0.0, 0.0])[:3]

    # current EEF poses for integration
    left_ee  = list(data["left_ee_pose"])   # [pos3, quat4]
    right_ee = list(data["right_ee_pose"])

    wide_actions = []
    for i in range(use_chunk):
        act = action_chunk[i]  # 14D

        # left arm: dpos[0:3], drotvec[3:6], gripper[6]
        left_dpos    = act[0:3]
        left_drotvec = act[3:6]
        left_gripper = float(act[6])

        # right arm: dpos[7:10], drotvec[10:13], gripper[13]
        right_dpos    = act[7:10]
        right_drotvec = act[10:13]
        right_gripper = float(act[13])

        # integrate delta onto current pose
        left_target  = integrate_delta_action(left_ee,  left_dpos,  left_drotvec)
        right_target = integrate_delta_action(right_ee, right_dpos, right_drotvec)

        # update current for next step (recurrent integration)
        left_ee  = left_target.tolist()
        right_ee = right_target.tolist()

        # pack into wide vector
        # slots [0:6]  = left  pos3 + rotvec3 (EE_POSE target)
        # slots [6:12] = right pos3 + rotvec3 (EE_POSE target)
        left_rotvec  = quat_xyzw_to_rotvec(left_target[3:7])
        right_rotvec = quat_xyzw_to_rotvec(right_target[3:7])

        wide = [0.0] * 20
        wide[0:3]   = left_target[:3].tolist()
        wide[3:6]   = left_rotvec.tolist()
        wide[6:9]   = right_target[:3].tolist()
        wide[9:12]  = right_rotvec.tolist()
        wide[12:15] = torso
        wide[15]    = left_gripper
        wide[16]    = right_gripper

        wide_actions.append(wide)

    return wide_actions


# ── main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="task/real/last0_deploy")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.deploy.deploy_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("deploy_last0.py  START")
    logger.info(f"  model_dir  : {cfg.last0.model_dir}")
    logger.info(f"  stats_path : {cfg.last0.stats_path}")
    logger.info(f"  robot      : {cfg.deploy.host}:{cfg.deploy.port}")
    logger.info(f"  use_chunk  : {cfg.deploy.use_chunk}")
    logger.info(f"  prompt     : {cfg.deploy.prompt}")
    logger.info("=" * 60)

    # ── check paths ───────────────────────────────────────────────────────────
    if not Path(cfg.last0.model_dir).exists():
        logger.error(f"[FATAL] model_dir not found: {cfg.last0.model_dir}")
        sys.exit(1)
    if not Path(cfg.last0.stats_path).exists():
        logger.error(f"[FATAL] stats_path not found: {cfg.last0.stats_path}")
        sys.exit(1)
    logger.info("[OK] checkpoint paths exist")

    # ── check CUDA ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        logger.error("[FATAL] No CUDA device found")
        sys.exit(1)
    logger.info(f"[OK] CUDA available: {torch.cuda.get_device_name(cfg.last0.cuda)}")

    # ── load model ────────────────────────────────────────────────────────────
    logger.info("Loading VLChatProcessor ... (may take 10-30s)")
    vl_chat_processor = VLChatProcessor.from_pretrained(
        cfg.last0.model_dir, trust_remote_code=True
    )
    logger.info("[OK] VLChatProcessor loaded")

    logger.info("Loading LaST0 model weights ... (may take 30-120s)")
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        cfg.last0.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        flow=True,
        action_dim=cfg.last0.action_dim,
        action_chunk=cfg.last0.action_chunk,
        use_pointcloud=False,
        use_latent=cfg.last0.use_latent,
        ignore_mismatched_sizes=True,
    )
    device = f"cuda:{cfg.last0.cuda}"
    logger.info(f"Moving model to {device} ...")
    vl_gpt = vl_gpt.to(device).eval()
    logger.info(f"[OK] Model on {device}")

    action_tokenizer = ActionTokenizer(vl_chat_processor.tokenizer, need_to_sub=3)

    logger.info(f"Loading stats from {cfg.last0.stats_path}")
    statistic = load_stats(cfg.last0.stats_path)
    logger.info(f"[OK] Stats loaded  action_dim={statistic['action_min'].shape}")

    # ── build a minimal cfg-like object for get_action() ─────────────────────
    @dataclass
    class InferCfg:
        cuda: int = cfg.last0.cuda
        latent_size: int = cfg.last0.latent_size
        use_latent: bool = cfg.last0.use_latent
        use_proprio: bool = cfg.last0.use_proprio
        num_open_loop_steps: int = cfg.last0.action_chunk

    infer_cfg = InferCfg()

    # ── HTTP session ──────────────────────────────────────────────────────────
    session = requests.Session()
    session.trust_env = False

    logger.info("=" * 60)
    logger.info("All models loaded. Starting deploy loop.")
    logger.info(f"  target: http://{cfg.deploy.host}:{cfg.deploy.port}")
    logger.info("  Make sure the robot HTTP server is running!")
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 60)

    data = None
    _conn_fail_count = 0
    try:
        while True:
            t0 = time.time()

            # ── 1. get state ──────────────────────────────────────────────────
            try:
                resp = session.get(
                    f"http://{cfg.deploy.host}:{cfg.deploy.port}/get_state",
                    timeout=2,
                )
            except requests.exceptions.RequestException as e:
                _conn_fail_count += 1
                logger.warning(
                    f"[{_conn_fail_count}] get_state failed: {e}  "
                    f"(target: http://{cfg.deploy.host}:{cfg.deploy.port}/get_state)"
                )
                if _conn_fail_count == 1:
                    logger.warning("  -> Is the robot server started? Check IP/port.")
                time.sleep(cfg.deploy.loop_interval)
                continue

            if resp.status_code != 200:
                logger.warning(f"get_state HTTP {resp.status_code}  body={resp.text[:200]}")
                time.sleep(cfg.deploy.loop_interval)
                continue

            _conn_fail_count = 0
            data = resp.json()
            latency_ms = (time.time() - t0) * 1000
            logger.info(f"get_state OK  latency={latency_ms:.1f}ms  keys={list(data.keys())}")

            # ── 2. decode images ──────────────────────────────────────────────
            try:
                head_pil       = decode_image_to_pil(data["image_head_base64"])
                left_wrist_pil = decode_image_to_pil(data["image_wrist_left_base64"])
                right_wrist_pil= decode_image_to_pil(data["image_wrist_right_base64"])
            except Exception as e:
                logger.error(f"Image decode failed: {e}")
                time.sleep(cfg.deploy.loop_interval)
                continue

            # ── 3. build state ────────────────────────────────────────────────
            try:
                state_16d = build_state_16d(data)
            except KeyError as e:
                logger.error(
                    f"Missing field in get_state response: {e}. "
                    "Ensure the robot server exposes left_ee_pose / right_ee_pose / "
                    "left_gripper / right_gripper."
                )
                time.sleep(cfg.deploy.loop_interval)
                continue

            # ── 4. inference ──────────────────────────────────────────────────
            try:
                # slow = head, fast = [left_wrist, right_wrist]
                slow_image = [head_pil]
                fast_image = [left_wrist_pil, right_wrist_pil]

                action_list = get_action(
                    cfg=infer_cfg,
                    statistic=statistic,
                    action_tokenizer=action_tokenizer,
                    vl_chat_processor=vl_chat_processor,
                    task_description=cfg.deploy.prompt,
                    vl_gpt=vl_gpt,
                    fast_image=fast_image,
                    slow_image=slow_image,
                    state=state_16d.tolist() if infer_cfg.use_proprio else None,
                )
            except Exception as e:
                logger.error(f"Inference failed: {e}", exc_info=True)
                time.sleep(cfg.deploy.loop_interval)
                continue

            # get_action returns list of length action_chunk; stack to array
            action_chunk = np.array(action_list, dtype=np.float32)  # [chunk, 14]
            logger.info(
                f"action_chunk shape={action_chunk.shape}  "
                f"left_dpos={action_chunk[0, :3]}  "
                f"right_dpos={action_chunk[0, 7:10]}"
            )

            # ── 5. build wide action and send ─────────────────────────────────
            use_chunk = min(cfg.deploy.use_chunk, len(action_chunk))
            wide_actions = build_wide_action_chunk(action_chunk, data, use_chunk)

            try:
                resp2 = requests.post(
                    f"http://{cfg.deploy.host}:{cfg.deploy.port}/apply_action",
                    json={"action_chunk": wide_actions},
                )
                if resp2.status_code == 200:
                    logger.info(f"apply_action OK  {resp2.json()}")
                else:
                    logger.warning(f"apply_action HTTP {resp2.status_code}")
            except Exception as e:
                logger.error(f"apply_action failed: {e}")

            time.sleep(cfg.deploy.loop_interval)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
        # send a safe stop: hold current pose
        if data is not None:
            try:
                torso = data.get("qpos_torso", [0.0, 0.0, 0.0])[:3]
                safe = [0.0] * 20
                safe[12:15] = torso
                safe[15] = float(data.get("left_gripper", 0.0))
                safe[16] = float(data.get("right_gripper", 0.0))
                requests.post(
                    f"http://{cfg.deploy.host}:{cfg.deploy.port}/apply_action",
                    json={"action_chunk": [safe] * 10},
                )
            except Exception:
                pass
    finally:
        session.close()
        logger.info("Session closed")


if __name__ == "__main__":
    main()
