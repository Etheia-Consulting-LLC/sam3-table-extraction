import copy
from pathlib import Path

import modal

from sam3_table.training_config import SAM3LoRAConfig


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (mutates *base*)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def build_sweep_configs(
    base_config: SAM3LoRAConfig,
    global_overrides: dict | None = None,
) -> list[dict]:
    """Generate a list of config dicts, each with different hyperparameters.

    Parameters
    ----------
    base_config:
        The base configuration loaded from YAML.
    global_overrides:
        Overrides applied to *every* run in the sweep.  Supports nested
        keys, e.g. ``{"training": {"num_epochs": 1, "data": {"sample_percent": 5.0}}}``.
        Per-run values in ``sweep_params`` take precedence over globals.
    """
    configs = []

    sweep_params = [
        {"lora": {"rank": 8, "alpha": 16}, "training": {"learning_rate": 1e-4}},
        {"lora": {"rank": 8, "alpha": 16, "apply_to_geometry_encoder": True, "apply_to_detr_encoder": True, "apply_to_detr_decoder": True, "apply_to_mask_decoder": True}, "training": {"learning_rate": 1e-4}},
        {"lora": {"rank": 16, "alpha": 32}, "training": {"learning_rate": 5e-5}},
        {"lora": {"rank": 16, "alpha": 32}, "training": {"learning_rate": 1e-4}},
        {"lora": {"rank": 32, "alpha": 64}, "training": {"learning_rate": 1e-4}},
        {"lora": {"rank": 32, "alpha": 64}, "training": {"learning_rate": 5e-5}},
    ]

    for overrides in sweep_params:
        base_dict = copy.deepcopy(base_config.model_dump(mode="json"))
        if global_overrides:
            _deep_merge(base_dict, global_overrides)
        _deep_merge(base_dict, overrides)
        configs.append(base_dict)

    return configs


if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "sam3_table" / "testSamples" / "full_lora_config.yaml"
    base_config = SAM3LoRAConfig.from_yaml(path)

    global_overrides = {
        "training": {"num_epochs": 1, "data": {"sample_percent": 5.0}},
    }

    configs = build_sweep_configs(base_config, global_overrides=global_overrides)

    sweep_fn = modal.Function.from_name("training-sam3", "run_sweep")
    call = sweep_fn.spawn(configs)
    print(f"Sweep launched with {len(configs)} configs. Function call ID: {call.object_id}")
    print("Interrupted runs are resumed automatically.")
    print("You can safely shut down this machine. Results will be saved to artifacts-vol.")
