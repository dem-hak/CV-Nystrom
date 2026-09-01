from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class UNetConfig:
    """Small config (~38M params) for single-GPU / 3090."""
    in_ch: int = 3
    out_ch: int = 3
    base_ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.1
    num_heads: int = 4
    image_size: int = 32
    num_classes: int = 10


@dataclass
class MultiResDriftConfig:
    """Config for multi-resolution drift experiments."""

    # Training
    batch_size: int = 128
    lr: float = 2e-4
    ema_decay: float = 0.9999
    max_grad_norm: float = 2.0
    total_steps: int = 50_000
    temperatures: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.2])
    dataset: str = "cifar10"

    # Loss selection.  "standard" preserves the original standalone path;
    # "ot" enables the W-Flow debiased Sinkhorn loss below.
    loss_mode: str = "standard"

    # Optional class conditioning.  It is off by default so existing
    # unconditional experiments keep the exact same model/training path.
    class_conditional: bool = False
    num_classes: int = 10

    # W-Flow OT settings.  The smaller defaults are intended for a single
    # GPU while retaining the repo's debiased multi-R update.
    ot_R_list: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.2])
    ot_sinkhorn_num_iter: int = 20
    ot_sinkhorn_stop_thr: float = 1e-4
    # The default matches the repo's OT configs: resampled negatives are
    # independent, so the generated-to-negative diagonal is not masked.
    ot_disable_diag_mask: bool = True
    ot_batch_sinkhorn: bool = True
    ot_use_quadratic_cost: bool = True
    # BIG CHANGE: opt-in second-order OT and landmark acceleration.  Keeping
    # these disabled preserves the original Sinkhorn training behavior.
    ot_use_newton: bool = False
    ot_nystrom_landmarks: int = 0
    ot_nystrom_ridge: float = 1e-4
    ot_newton_max_iter: int = 5
    ot_newton_stop_thr: float = 1e-6
    # Zero selects a VRAM-aware chunk.  Increase only after checking peak
    # memory on the specific RTX 6000/model/feature-resolution combination.
    ot_plan_chunk_size: int = 0
    ot_gen_per_label: int = 4
    ot_pos_per_label: int = 16
    ot_neg_per_label: int = 16
    ot_class_conditional: bool = True
    ot_use_new_cfg: bool = False
    ot_cfg_min: float = 1.0
    ot_cfg_max: float = 4.0
    ot_neg_cfg_pw: float = 1.0
    ot_no_cfg_frac: float = 0.0
    ot_resample_neg: bool = True

    # Multi-res encoder
    encoder: str = "dinov3"
    pool_size: int = 4  # spatial pool target per stage

    # Core Nyström Parameters
    nystrom_landmarks_per_class: int = 328    # Optimal ratio (0.1024). Paper Fig 5 shows ~512 is sweet spot.
    nystrom_ridge: float = 1e-4
    nystrom_landmark_seed: int = 0

    # Logging (Standard)
    log_every: int = 100
    sample_every: int = 5_000
    save_every: int = 10_000
