from __future__ import annotations

import os
import modal

from sam3_table.training_config import SAM3LoRAConfig

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("triton")
    .add_local_python_source("sam3_train")
)

app = modal.App(name="training-sam3", image = image)

@app.function(gpu="A100", image=image)
def train_sam3(
    config: SAM3LoRAConfig,
    device: list[int] | None = None,
) -> None:

    from train_sam3_lora_native import SAM3TrainerNative
    

    if device is None:
        device = [0]

    multi_gpu = len(device) > 1 and "LOCAL_RANK" in os.environ

    if not multi_gpu and len(device) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0])
        print(f"Using single GPU: {device[0]}")

    trainer = SAM3TrainerNative(config, multi_gpu=multi_gpu)
    trainer.train()

    

    