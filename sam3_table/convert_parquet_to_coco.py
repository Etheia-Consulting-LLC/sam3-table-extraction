"""Convert a HuggingFace-style parquet dataset into COCO format.

Reads a parquet file whose rows contain:
  - image: {bytes: <raw jpeg>, path: <filename>}
  - objects: {bbox: [array([x1,y1,x2,y2]), ...], categories: "table"}

Produces:
  - A directory of extracted JPEG images
  - A COCO-format JSON annotation file compatible with coco_schema.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


def convert_parquet_to_coco(
    parquet_path: str | Path,
    output_dir: str | Path,
    annotation_filename: str = "annotations.json",
    images_subdir: str = "images",
) -> Path:
    """Convert a parquet dataset to COCO images + annotation JSON.

    Parameters
    ----------
    parquet_path : path to the .parquet file
    output_dir : root directory for output (images/ and annotations.json)
    annotation_filename : name of the output JSON file
    images_subdir : subdirectory under *output_dir* for extracted images

    Returns
    -------
    Path to the written annotation JSON file.
    """
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    images_dir = output_dir / images_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    category_map: dict[str, int] = {}
    ann_id = 1

    for image_id, (_, row) in enumerate(df.iterrows(), start=1):
        img_meta = row["image"]
        objects = row["objects"]

        img_bytes: bytes = img_meta["bytes"]
        file_name: str = img_meta["path"]

        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size

        dest = images_dir / file_name
        if not dest.exists():
            dest.write_bytes(img_bytes)

        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        bboxes = objects["bbox"]
        cat_value = objects["categories"]
        categories_list = (
            cat_value if isinstance(cat_value, list) else [cat_value] * len(bboxes)
        )

        for bbox_arr, cat_name in zip(bboxes, categories_list):
            if cat_name not in category_map:
                category_map[cat_name] = len(category_map) + 1

            x1, y1, x2, y2 = bbox_arr.tolist()
            w = x2 - x1
            h = y2 - y1
            area = w * h

            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_map[cat_name],
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(area, 2),
                    "segmentation": None,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_categories = [
        {"id": cat_id, "name": cat_name}
        for cat_name, cat_id in sorted(category_map.items(), key=lambda x: x[1])
    ]

    coco_dataset = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    annotation_path = output_dir / annotation_filename
    with open(annotation_path, "w") as f:
        json.dump(coco_dataset, f)

    print(f"Extracted {len(coco_images)} images to {images_dir}")
    print(f"Generated {len(coco_annotations)} annotations across {len(coco_categories)} categories")
    print(f"Annotation file: {annotation_path}")

    return annotation_path


if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else "sam3_table/testSamples/train-00000-of-00001.parquet"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "sam3_table/table_dataset"
    convert_parquet_to_coco(parquet, out_dir)
