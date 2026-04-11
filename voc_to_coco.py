from __future__ import annotations

import argparse
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


def _iter_voc_xml_inputs(input_path: Path) -> Iterable[tuple[str, bytes]]:
    """Yield `(source_name, xml_bytes)` pairs from a directory or tar archive."""
    if input_path.is_dir():
        for xml_path in sorted(input_path.rglob("*.xml")):
            yield str(xml_path), xml_path.read_bytes()
        return

    suffixes = input_path.suffixes
    if suffixes[-2:] == [".tar", ".gz"] or suffixes[-1:] == [".tgz"]:
        with tarfile.open(input_path, "r:*") as tar:
            for member in sorted(
                (member for member in tar.getmembers() if member.isfile()),
                key=lambda item: item.name,
            ):
                if not member.name.lower().endswith(".xml"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                yield member.name, extracted.read()
        return

    raise ValueError(
        f"Unsupported input path: {input_path}. "
        "Expected a directory of XML files or a .tar.gz/.tgz archive."
    )


def _text_or_none(node: ET.Element | None, tag: str) -> str | None:
    if node is None:
        return None
    child = node.find(tag)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value or None


def _resolve_file_name(
    root: ET.Element,
    source_name: str,
    fallback_image_extension: str,
) -> str:
    filename = _text_or_none(root, "filename")
    if filename:
        return Path(filename).name

    path_text = _text_or_none(root, "path")
    if path_text:
        return Path(path_text).name

    return f"{Path(source_name).stem}{fallback_image_extension}"


def _load_category_map(category_map_path: Path | None) -> dict[str, str]:
    if category_map_path is None:
        return {}

    data = json.loads(category_map_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Category map JSON must be an object, e.g. {'table rotated': 'table'}")

    normalized: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Category map keys and values must both be strings")
        normalized[key.strip()] = value.strip()
    return normalized


def convert_voc_to_coco(
    input_path: str | Path,
    output_json: str | Path,
    *,
    single_category_name: str | None = None,
    category_map_path: str | Path | None = None,
    fallback_image_extension: str = ".jpg",
) -> dict:
    input_path = Path(input_path)
    output_json = Path(output_json)
    category_map = _load_category_map(Path(category_map_path) if category_map_path else None)

    if single_category_name is not None:
        single_category_name = single_category_name.strip()
        if not single_category_name:
            raise ValueError("single_category_name must not be empty")

    images: list[dict] = []
    annotations: list[dict] = []
    categories: list[dict] = []
    category_name_to_id: dict[str, int] = {}

    next_image_id = 1
    next_annotation_id = 1

    for source_name, xml_bytes in _iter_voc_xml_inputs(input_path):
        root = ET.fromstring(xml_bytes)

        size_node = root.find("size")
        width_text = _text_or_none(size_node, "width")
        height_text = _text_or_none(size_node, "height")
        if width_text is None or height_text is None:
            raise ValueError(f"Missing image size in {source_name}")

        width = int(float(width_text))
        height = int(float(height_text))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {source_name}: width={width}, height={height}")

        file_name = _resolve_file_name(root, source_name, fallback_image_extension)

        current_image_id = next_image_id
        next_image_id += 1
        images.append(
            {
                "id": current_image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for obj in root.findall("object"):
            raw_name = _text_or_none(obj, "name")
            if raw_name is None:
                continue

            category_name = raw_name
            if category_name in category_map:
                category_name = category_map[category_name]
            if single_category_name is not None:
                category_name = single_category_name

            if category_name not in category_name_to_id:
                category_id = len(category_name_to_id) + 1
                category_name_to_id[category_name] = category_id
                categories.append({"id": category_id, "name": category_name})
            else:
                category_id = category_name_to_id[category_name]

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin_text = _text_or_none(bndbox, "xmin")
            ymin_text = _text_or_none(bndbox, "ymin")
            xmax_text = _text_or_none(bndbox, "xmax")
            ymax_text = _text_or_none(bndbox, "ymax")
            if None in (xmin_text, ymin_text, xmax_text, ymax_text):
                continue

            xmin = float(xmin_text)
            ymin = float(ymin_text)
            xmax = float(xmax_text)
            ymax = float(ymax_text)

            x = max(0.0, min(xmin, width))
            y = max(0.0, min(ymin, height))
            x2 = max(0.0, min(xmax, width))
            y2 = max(0.0, min(ymax, height))
            bbox_width = max(0.0, x2 - x)
            bbox_height = max(0.0, y2 - y)

            if bbox_width <= 0.0 or bbox_height <= 0.0:
                continue

            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": current_image_id,
                    "category_id": category_id,
                    "bbox": [x, y, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "segmentation": None,
                    "iscrowd": 0,
                }
            )
            next_annotation_id += 1

    if not images:
        raise ValueError(f"No VOC XML annotations found in {input_path}")

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco, indent=2))
    return coco


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML annotations to a COCO JSON file.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Directory containing VOC XML files, or a .tar.gz/.tgz archive.",
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Path to write the COCO JSON file.",
    )
    parser.add_argument(
        "--single-category-name",
        type=str,
        default=None,
        help="Force all objects into one category, e.g. 'table'.",
    )
    parser.add_argument(
        "--category-map",
        type=Path,
        default=None,
        help="Path to a JSON object mapping VOC category names to COCO category names.",
    )
    parser.add_argument(
        "--fallback-image-extension",
        type=str,
        default=".jpg",
        help="Used if an XML file has no <filename> or <path>. Default: .jpg",
    )
    args = parser.parse_args()

    coco = convert_voc_to_coco(
        input_path=args.input_path,
        output_json=args.output_json,
        single_category_name=args.single_category_name,
        category_map_path=args.category_map,
        fallback_image_extension=args.fallback_image_extension,
    )

    print(f"Saved COCO JSON to {args.output_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")


if __name__ == "__main__":
    main()
