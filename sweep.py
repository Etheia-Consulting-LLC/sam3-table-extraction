import copy
from pathlib import Path

from sam3_table.training_config import SAM3LoRAConfig
from sam3_table.cstone_train_sam3 import train_sam3, app


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

    with app.run():
        handles = []
        for i, config in enumerate(configs):
            handle = train_sam3.spawn(config)
            handles.append((i, config, handle))
            lora = config["lora"]
            lr = config["training"]["learning_rate"]
            print(f"Launched run {i}: rank={lora['rank']} alpha={lora['alpha']} lr={lr}")

        print(f"\nAll {len(handles)} runs launched in parallel. Waiting for results...\n")

        for i, config, handle in handles:
            result = handle.get()
            lora = config["lora"]
            lr = config["training"]["learning_rate"]
            print(f"--- Run {i} (rank={lora['rank']} alpha={lora['alpha']} lr={lr}) ---")
            print(f"  Timestamp:  {result['timestamp']}")
            print(f"  Output dir: {result['output_dir']}")
