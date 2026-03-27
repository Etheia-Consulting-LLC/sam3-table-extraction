from __future__ import annotations

import os
from pathlib import Path

import modal

from sam3_table.training_config import SAM3LoRAConfig
from sam3_table.coco_schema import COCODataset

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("triton")
    .add_local_python_source("sam3_table")
)

app = modal.App(name="training-sam3", image = image)

data_vol = modal.Volume.from_name("data-vol", create_if_missing=True)
artifacts_vol = modal.Volume.from_name("artifacts-vol", create_if_missing=True)

MODAL_DATA_DIR = "/data"
MODAL_ARTIFACTS_DIR = "/artifacts"
LOCAL_IMAGE_DIR = Path(__file__).resolve().parent / "table_dataset" / "images"


def upload_image_directory(local_image_dir: str | Path = LOCAL_IMAGE_DIR) -> None:
    """Upload the local training image directory into the mounted Modal volume."""
    local_image_dir = Path(local_image_dir)
    with data_vol.batch_upload() as batch:
        batch.put_directory(str(local_image_dir), "/")


@app.function(
    gpu="H200",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODAL_DATA_DIR: data_vol, MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600,
)
def train_sam3(
    config_dict: dict,
    train_coco_dataset: COCODataset | None = None,
    val_coco_dataset: COCODataset | None = None,
    test_coco_dataset: COCODataset | None = None,
    device: list[int] | None = None,
) -> None:

    from sam3_table.train_sam3_lora_native import SAM3TrainerNative

    config = SAM3LoRAConfig.model_validate(config_dict)
    config.output.output_dir = f"{MODAL_ARTIFACTS_DIR}/{config.output.output_dir}"

    if device is None:
        device = [0]

    multi_gpu = len(device) > 1 and "LOCAL_RANK" in os.environ

    if not multi_gpu and len(device) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0])
        print(f"Using single GPU: {device[0]}")

    trainer = SAM3TrainerNative(
        config,
        train_coco_dataset=train_coco_dataset,
        val_coco_dataset=val_coco_dataset,
        test_coco_dataset=test_coco_dataset,
        multi_gpu=multi_gpu,
    )
    trainer.train()

    artifacts_vol.commit()
    print(f"Training artifacts committed to 'training-artifacts' volume.")
