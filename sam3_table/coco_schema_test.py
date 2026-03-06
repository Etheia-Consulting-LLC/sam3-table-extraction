"""Golden tests for COCODataset schema parsed from small_coco/instances_train2017_small.json."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from coco_schema import (
    COCOAnnotation,
    COCOCategory,
    COCODataset,
    COCOImage,
    RLESegmentation,
)

SAMPLE_JSON = Path(__file__).parent / "small_coco" / "instances_train2017_small.json"


@pytest.fixture
def dataset() -> COCODataset:
    return COCODataset.from_json(SAMPLE_JSON)


# ── Top-level structure ─────────────────────────────────────────────────────

class TestDatasetStructure:
    def test_image_count(self, dataset: COCODataset):
        assert len(dataset.images) == 2

    def test_annotation_count(self, dataset: COCODataset):
        assert len(dataset.annotations) == 3

    def test_category_count(self, dataset: COCODataset):
        assert len(dataset.categories) == 2


# ── Images ──────────────────────────────────────────────────────────────────

class TestImages:
    def test_first_image(self, dataset: COCODataset):
        img = dataset.images[0]
        assert img.id == 0
        assert img.file_name == "img001.jpg"
        assert img.height == 480
        assert img.width == 640

    def test_second_image(self, dataset: COCODataset):
        img = dataset.images[1]
        assert img.id == 1
        assert img.file_name == "img002.png"
        assert img.height == 1024
        assert img.width == 768


# ── Annotations ─────────────────────────────────────────────────────────────

class TestAnnotations:
    def test_polygon_annotation(self, dataset: COCODataset):
        """Annotation id=1: polygon segmentation on image 0."""
        ann = dataset.annotations[0]
        assert ann.id == 1
        assert ann.image_id == 0
        assert ann.category_id == 1
        assert ann.bbox == [100.0, 150.0, 200.0, 120.0]
        assert ann.area == pytest.approx(24000.0)
        assert ann.iscrowd == 0

        assert isinstance(ann.segmentation, list)
        assert len(ann.segmentation) == 1
        assert ann.segmentation[0] == [
            100.0, 150.0, 300.0, 150.0, 300.0, 270.0, 100.0, 270.0
        ]

    def test_rle_annotation(self, dataset: COCODataset):
        """Annotation id=2: RLE segmentation on image 0."""
        ann = dataset.annotations[1]
        assert ann.id == 2
        assert ann.image_id == 0
        assert ann.category_id == 2
        assert ann.bbox == [50.0, 60.0, 80.0, 90.0]
        assert ann.area == pytest.approx(7200.0)
        assert ann.iscrowd == 0

        assert isinstance(ann.segmentation, RLESegmentation)
        assert ann.segmentation.counts == "abc123"
        assert ann.segmentation.size == [480, 640]

    def test_bbox_only_annotation(self, dataset: COCODataset):
        """Annotation id=3: no segmentation, iscrowd=1, on image 1."""
        ann = dataset.annotations[2]
        assert ann.id == 3
        assert ann.image_id == 1
        assert ann.category_id == 1
        assert ann.bbox == [10.0, 20.0, 50.0, 40.0]
        assert ann.area == pytest.approx(2000.0)
        assert ann.iscrowd == 1
        assert ann.segmentation is None


# ── Categories ──────────────────────────────────────────────────────────────

class TestCategories:
    def test_first_category(self, dataset: COCODataset):
        assert dataset.categories[0].id == 1
        assert dataset.categories[0].name == "table"

    def test_second_category(self, dataset: COCODataset):
        assert dataset.categories[1].id == 2
        assert dataset.categories[1].name == "cell"


# ── Round-trip serialization ────────────────────────────────────────────────

class TestRoundTrip:
    def test_json_round_trip(self, dataset: COCODataset):
        json_str = dataset.model_dump_json()
        reloaded = COCODataset.model_validate_json(json_str)
        assert reloaded == dataset

    def test_file_round_trip(self, dataset: COCODataset, tmp_path: Path):
        import json

        out_path = tmp_path / "roundtrip.json"
        with open(out_path, "w") as f:
            json.dump(dataset.model_dump(mode="json"), f)

        reloaded = COCODataset.from_json(out_path)
        assert reloaded == dataset


# ── Validation edge cases ───────────────────────────────────────────────────

class TestValidation:
    def test_image_height_must_be_positive(self):
        with pytest.raises(ValidationError):
            COCOImage(id=0, file_name="a.jpg", height=0, width=100)

    def test_image_width_must_be_positive(self):
        with pytest.raises(ValidationError):
            COCOImage(id=0, file_name="a.jpg", height=100, width=0)

    def test_bbox_must_have_four_elements(self):
        with pytest.raises(ValidationError):
            COCOAnnotation(
                id=0, image_id=0, category_id=1,
                bbox=[10.0, 20.0, 30.0],
                area=600.0,
            )

    def test_bbox_negative_width_rejected(self):
        with pytest.raises(ValidationError, match="width and height must be >= 0"):
            COCOAnnotation(
                id=0, image_id=0, category_id=1,
                bbox=[10.0, 20.0, -5.0, 30.0],
                area=0.0,
            )

    def test_bbox_negative_height_rejected(self):
        with pytest.raises(ValidationError, match="width and height must be >= 0"):
            COCOAnnotation(
                id=0, image_id=0, category_id=1,
                bbox=[10.0, 20.0, 30.0, -5.0],
                area=0.0,
            )

    def test_area_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            COCOAnnotation(
                id=0, image_id=0, category_id=1,
                bbox=[10.0, 20.0, 30.0, 40.0],
                area=-1.0,
            )

    def test_iscrowd_must_be_0_or_1(self):
        with pytest.raises(ValidationError):
            COCOAnnotation(
                id=0, image_id=0, category_id=1,
                bbox=[10.0, 20.0, 30.0, 40.0],
                area=1200.0, iscrowd=2,
            )

    def test_rle_size_must_have_two_elements(self):
        with pytest.raises(ValidationError):
            RLESegmentation(counts="abc", size=[480])

    def test_duplicate_image_ids_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate image ids"):
            COCODataset(
                images=[
                    COCOImage(id=0, file_name="a.jpg", height=100, width=100),
                    COCOImage(id=0, file_name="b.jpg", height=100, width=100),
                ],
                annotations=[],
                categories=[COCOCategory(id=1, name="x")],
            )

    def test_duplicate_annotation_ids_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate annotation ids"):
            COCODataset(
                images=[COCOImage(id=0, file_name="a.jpg", height=100, width=100)],
                annotations=[
                    COCOAnnotation(id=1, image_id=0, category_id=1, bbox=[0, 0, 10, 10], area=100),
                    COCOAnnotation(id=1, image_id=0, category_id=1, bbox=[5, 5, 10, 10], area=100),
                ],
                categories=[COCOCategory(id=1, name="x")],
            )

    def test_segmentation_absent_is_valid(self):
        ann = COCOAnnotation(
            id=0, image_id=0, category_id=1,
            bbox=[0.0, 0.0, 10.0, 10.0],
            area=100.0,
        )
        assert ann.segmentation is None

    def test_empty_dataset_is_valid(self):
        ds = COCODataset(images=[], annotations=[], categories=[])
        assert len(ds.images) == 0
        assert len(ds.annotations) == 0
        assert len(ds.categories) == 0
