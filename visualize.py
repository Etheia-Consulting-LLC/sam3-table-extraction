from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Any

try:
    from sam3_table.coco_schema import COCODataset
    from sam3_table.cstone_train_sam3 import (
        MODAL_DATA_DIR,
        app,
        data_vol,
        image,
        infer_sam3,
    )
except Exception:  # pragma: no cover - optional when Modal dependencies are unavailable
    COCODataset = None
    app = None
    image = None
    data_vol = None
    MODAL_DATA_DIR = None
    infer_sam3 = None

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]


def _color_for_category(category_id: int) -> str:
    return COLORS[category_id % len(COLORS)]


def _render_boxes_on_image(
    pil_image: Any,
    boxes: list[dict[str, Any]],
    category_map: dict[int, str],
) -> bytes:
    """Draw boxes on *pil_image* and return the result as PNG bytes.

    Each entry in *boxes* must have at minimum ``bbox`` ([x, y, w, h]) and
    ``category_id``.  An optional ``score`` field is shown in the label.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(pil_image)

    for box in boxes:
        x, y, w, h = box["bbox"]
        cat_id = box["category_id"]
        color = _color_for_category(cat_id)
        label = category_map.get(cat_id, str(cat_id))
        if "score" in box:
            label = f"{label} {box['score']:.2f}"

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x, y - 4, label,
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
        )

    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _extract_timestamp_from_weights_path(weights_path: str | Path) -> str:
    weights_path = Path(weights_path)
    timestamp_pattern = re.compile(r"^\d{8}-\d{6}$")

    for path_part in reversed(weights_path.parts):
        if timestamp_pattern.fullmatch(path_part):
            return path_part

    raise ValueError(
        f"Could not extract a run timestamp from weights path: {weights_path}"
    )


def visualize_model_predictions(
    weights_path: str | Path,
    image_paths: list[str | Path],
    *,
    score_threshold: float = 0.25,
    query_text: str = "table",
) -> list[dict[str, Any]]:
    """Run Modal inference for the given weights path and render predictions."""
    if app is None or infer_sam3 is None:
        raise RuntimeError("Modal inference is unavailable in this environment.")

    timestamp = _extract_timestamp_from_weights_path(weights_path)
    requested_image_refs = [str(image_path) for image_path in image_paths]
    resolved_paths = [Path(image_ref) for image_ref in requested_image_refs]
    image_names = [path.name for path in resolved_paths]

    image_bytes_list: list[bytes | None] = [None] * len(resolved_paths)
    volume_image_refs: list[str] = []
    volume_indices: list[int] = []

    for idx, (image_ref, path) in enumerate(zip(requested_image_refs, resolved_paths)):
        if path.exists():
            image_bytes_list[idx] = path.read_bytes()
        else:
            volume_image_refs.append(image_ref)
            volume_indices.append(idx)

    if volume_image_refs:
        with app.run():
            volume_image_bytes = load_images_from_volume.remote(
                image_names=volume_image_refs
            )
        for idx, image_bytes in zip(volume_indices, volume_image_bytes):
            image_bytes_list[idx] = image_bytes

    finalized_image_bytes = [image_bytes for image_bytes in image_bytes_list if image_bytes is not None]

    with app.run():
        inference_results = infer_sam3.remote(
            timestamp=timestamp,
            image_bytes_list=finalized_image_bytes,
            image_names=image_names,
            query_text=query_text,
            score_threshold=score_threshold,
        )
        return visualize_predictions.remote(
            inference_results=inference_results,
            image_bytes_list=finalized_image_bytes,
            category_map={1: query_text},
        )


if app is not None and image is not None and data_vol is not None and MODAL_DATA_DIR is not None:
    @app.function(
        image=image,
        volumes={MODAL_DATA_DIR: data_vol},
        timeout=600,
    )
    def load_images_from_volume(image_names: list[str]) -> list[bytes]:
        image_bytes_list: list[bytes] = []
        for image_name in image_names:
            image_path = Path(MODAL_DATA_DIR) / image_name
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found in Modal data volume: {image_name}"
                )
            image_bytes_list.append(image_path.read_bytes())
        return image_bytes_list


    @app.function(
        image=image,
        volumes={MODAL_DATA_DIR: data_vol},
        timeout=600,
    )
    def visualize_ground_truth(
        coco_dataset: COCODataset,
        image_ids: list[int] | None = None,
        max_images: int = 9,
    ) -> list[dict[str, Any]]:
        """Render ground-truth COCO boxes on images stored in the data volume.

        Returns a list of dicts, each with ``image_name`` (str) and
        ``png_bytes`` (bytes).
        """
        from PIL import Image as PILImage

        category_map = {cat.id: cat.name for cat in coco_dataset.categories}

        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco_dataset.annotations:
            anns_by_image.setdefault(ann.image_id, []).append(
                {"bbox": ann.bbox, "category_id": ann.category_id}
            )

        images = coco_dataset.images
        if image_ids is not None:
            id_set = set(image_ids)
            images = [img for img in images if img.id in id_set]
        images = images[:max_images]

        results: list[dict[str, Any]] = []
        for img_entry in images:
            img_path = Path(MODAL_DATA_DIR) / img_entry.file_name
            pil_img = PILImage.open(img_path).convert("RGB")
            png_bytes = _render_boxes_on_image(
                pil_img,
                anns_by_image.get(img_entry.id, []),
                category_map,
            )
            results.append({"image_name": img_entry.file_name, "png_bytes": png_bytes})

        return results


    @app.function(
        image=image,
        timeout=600,
    )
    def visualize_predictions(
        inference_results: dict[str, Any],
        image_bytes_list: list[bytes],
        category_map: dict[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Render predicted boxes on the original images.

        *inference_results* is the dict returned by ``infer_sam3``.
        *image_bytes_list* must be the same images (in the same order) that were
        passed to ``infer_sam3``.

        Returns a list of dicts, each with ``image_name`` (str) and
        ``png_bytes`` (bytes).
        """
        from PIL import Image as PILImage

        if category_map is None:
            category_map = {1: "table"}

        results: list[dict[str, Any]] = []
        for entry, raw_bytes in zip(inference_results["results"], image_bytes_list):
            pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
            png_bytes = _render_boxes_on_image(
                pil_img,
                entry["predictions"],
                category_map,
            )
            results.append({"image_name": entry["image_name"], "png_bytes": png_bytes})

        return results
else:
    def load_images_from_volume(*args: Any, **kwargs: Any) -> list[bytes]:
        raise RuntimeError("Modal image loading is unavailable in this environment.")


    def visualize_ground_truth(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("Modal visualization is unavailable in this environment.")


    def visualize_predictions(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("Modal visualization is unavailable in this environment.")


# ---------------------------------------------------------------------------
# Local helpers for saving / displaying the images returned by Modal
# ---------------------------------------------------------------------------

def save_results(results: list[dict[str, Any]], save_dir: Path) -> None:
    """Write annotated PNG images returned by a Modal function to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for item in results:
        out_path = save_dir / Path(item["image_name"]).with_suffix(".png").name
        out_path.write_bytes(item["png_bytes"])
        print(f"Saved: {out_path}")


def show_results(results: list[dict[str, Any]]) -> None:
    """Display annotated images returned by a Modal function with matplotlib."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    cols = min(3, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows * cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)

    for idx, item in enumerate(results):
        img = PILImage.open(io.BytesIO(item["png_bytes"]))
        axes_list[idx].imshow(img)
        axes_list[idx].set_title(item["image_name"], fontsize=10)
        axes_list[idx].set_axis_off()

    for idx in range(len(results), len(axes_list)):
        axes_list[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize ground truth on Modal or local model predictions.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("sam3_table/table_dataset/annotations.json"),
        help="Path to local COCO annotations JSON file.",
    )
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific image IDs to visualize (default: first N).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=9,
        help="Maximum number of images to show (default: 9).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to LoRA weights. This triggers Modal inference mode.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=None,
        help="One or more local images to run through Modal inference.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Score threshold for model prediction mode.",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        default="table",
        help="Text query to use for model prediction mode.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save annotated images here instead of displaying.",
    )
    args = parser.parse_args()

    if args.weights is not None:
        if not args.images:
            parser.error("--images is required when --weights is provided.")
        results = visualize_model_predictions(
            weights_path=args.weights,
            image_paths=args.images,
            score_threshold=args.score_threshold,
            query_text=args.query_text,
        )
    else:
        if COCODataset is None or app is None:
            parser.error(
                "Modal dependencies are unavailable in this environment."
            )

        coco = COCODataset.from_json(args.annotations)

        with app.run():
            results = visualize_ground_truth.remote(
                coco,
                image_ids=args.image_ids,
                max_images=args.max_images,
            )

    if args.save_dir:
        save_results(results, args.save_dir)
    else:
        show_results(results)