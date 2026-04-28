"""A/B test: optimized training (bs=8, workers=8) with vs without mixed precision.

Both runs use the same LoRA (rank=4, alpha=8), same batch/worker settings, and
only differ on mixed_precision:

  Run A: batch_size=8, num_workers=8, grad_accum=1, mixed_precision=bf16
  Run B: same, mixed_precision=no (FP32 autocast disabled)

Usage:
    modal run sweep_test.py
"""

import copy
from pathlib import Path

from sam3_table.training_config import SAM3LoRAConfig
from sam3_table.cstone_train_sam3 import app, run_sweep, train_sam3


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base (mutates and returns base)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base

SHARED = {
    "lora": {"rank": 16, "alpha": 32},
    "training": {
        "num_epochs": 1,
        "learning_rate": 1e-4,
        "data": {"sample_percent": 1},
        "batch_size": 8,
        "val_batch_size": 16,
        "num_workers": 8,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
    },
    "output": {
        "output_dir": "outputs/ab_testing/top2_changes",
    },
}


def build_test_sweep_configs(base_config: SAM3LoRAConfig) -> list[dict]:
    sweep_params = [
        {**SHARED, "training": {**SHARED["training"]}},
    ]

    configs = []
    for overrides in sweep_params:
        base_dict = copy.deepcopy(base_config.model_dump(mode="json"))
        _deep_merge(base_dict, overrides)
        configs.append(base_dict)

    return configs


@app.local_entrypoint()
def main(
    fresh_run: bool = True,
    resume_output_dir: str = "",
):
    path = Path(__file__).resolve().parent / "sam3_table" / "testSamples" / "full_lora_config.yaml"
    base_config = SAM3LoRAConfig.from_yaml(path)

    configs = build_test_sweep_configs(base_config)

    labels = ["optimized + BF16"]
    print(f"Launching mixed-precision test ({len(configs)} config, 1 epoch)...")
    print(f"  Run: {labels[0]}")

    if resume_output_dir.strip():
        # Manual resume mode: explicitly continue a specific prior run directory.
        print(f"Manual resume requested from: {resume_output_dir}")
        result = train_sam3.remote(
            configs[0],
            fresh_run=False,
            resume_output_dir=resume_output_dir.strip(),
        )
        results = [result]
    else:
        results = run_sweep.remote(configs, fresh_run=fresh_run)

    for i, r in enumerate(results):
        label = labels[i] if i < len(labels) else f"run {i}"
        if "error" in r:
            print(f"  FAIL - {label}: {r['error']}")
        else:
            print(f"  OK   - {label}: {r['timestamp']}  {r['output_dir']}")
