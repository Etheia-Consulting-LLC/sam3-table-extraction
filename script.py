import modal
from pathlib import Path

from sam3_table.training_config import SAM3LoRAConfig

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "sam3_table" / "testSamples" / "full_lora_config.yaml"
    config = SAM3LoRAConfig.from_yaml(path)

    train_fn = modal.Function.from_name("training-sam3", "train_sam3")
    call = train_fn.spawn(config.model_dump(mode="json"))
    print(f"Training launched. Function call ID: {call.object_id}")
    print("Interrupted runs are resumed automatically.")
    print("You can safely shut down this machine. Results will be saved to artifacts-vol.")
