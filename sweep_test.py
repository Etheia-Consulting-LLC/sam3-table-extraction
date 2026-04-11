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
from sweep import _deep_merge

SHARED = {
    "lora": {"rank": 4, "alpha": 8},
    "training": {
        "num_epochs": 1,
        "learning_rate": 1e-4,
        "data": {"sample_percent": 1.0},
        "batch_size": 8,
        "num_workers": 8,
        "gradient_accumulation_steps": 1,
    },
    "output": {
        "output_dir": "outputs/ab_testing/mixed_precision",
    },
}


def build_test_sweep_configs(base_config: SAM3LoRAConfig) -> list[dict]:
    sweep_params = [
        {**SHARED, "training": {**SHARED["training"], "mixed_precision": "bf16"}},
        {**SHARED, "training": {**SHARED["training"], "mixed_precision": "no"}},
        {**SHARED, "training": {**SHARED["training"], "mixed_precision": "bf16"}},
        {**SHARED, "training": {**SHARED["training"], "mixed_precision": "no"}},
        {**SHARED, "training": {**SHARED["training"], "mixed_precision": "bf16"}},
        {**SHARED, "training": {**SHARED["training"], "mixed_precision": "no"}},
    ]

    configs = []
    for overrides in sweep_params:
        base_dict = copy.deepcopy(base_config.model_dump(mode="json"))
        _deep_merge(base_dict, overrides)
        configs.append(base_dict)

    return configs


@app.local_entrypoint()
def main():
    path = Path(__file__).resolve().parent / "sam3_table" / "testSamples" / "full_lora_config.yaml"
    base_config = SAM3LoRAConfig.from_yaml(path)

    configs = build_test_sweep_configs(base_config)

    labels = ["optimized + BF16", "optimized + FP32 (no MP)"]
    print(f"Launching mixed-precision A/B test ({len(configs)} configs, 1 epoch each)...")
    print(f"  Run A: {labels[0]}")
    print(f"  Run B: {labels[1]}")
    results = run_sweep.remote(configs, fresh_run=True)

    for i, r in enumerate(results):
        label = labels[i] if i < len(labels) else f"run {i}"
        if "error" in r:
            print(f"  FAIL - {label}: {r['error']}")
        else:
            print(f"  OK   - {label}: {r['timestamp']}  {r['output_dir']}")
