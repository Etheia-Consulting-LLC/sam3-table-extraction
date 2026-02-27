import modal
from sam3_table.training_config import SAM3LoRAConfig
from sam3_table.cstone_train_sam3 import train_sam3, app

if __name__ == "__main__":
    path = r"/Users/andre/Documents/Repos/sam3-table-extraction/sam3_table/testSamples/full_lora_config.yaml"
    config = SAM3LoRAConfig.from_yaml(path)
    

    with app.run():
        print(train_sam3.remote(config))