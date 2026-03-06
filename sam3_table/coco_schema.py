"""Pydantic models for the COCO annotation format used by the training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class COCOImage(BaseModel):
    """A single image entry in the COCO dataset."""

    id: int
    file_name: str
    height: int = Field(gt=0)
    width: int = Field(gt=0)


class RLESegmentation(BaseModel):
    """Run-length encoded segmentation mask."""

    counts: str | list[int]
    size: list[int] = Field(min_length=2, max_length=2)

    @field_validator("counts")
    @classmethod
    def counts_must_be_string_or_int_list(cls, v: str | list[int]) -> str | list[int]:
        if isinstance(v, list) and not all(isinstance(x, int) for x in v):
            raise ValueError("RLE counts list must contain only integers")
        return v


class COCOAnnotation(BaseModel):
    """A single object annotation in the COCO dataset.

    Segmentation can be:
      - polygon format:  list of coordinate lists  [[x1, y1, x2, y2, ...], ...]
      - RLE format:      {"counts": "...", "size": [h, w]} or {"counts": [1, 2, ...], "size": [h, w]}
      - absent:          None (bbox-only annotation)
    """

    id: int
    image_id: int
    category_id: int
    bbox: list[float] = Field(min_length=4, max_length=4, description="[x, y, width, height]")
    area: float = Field(ge=0)
    segmentation: Optional[Union[list[list[float]], RLESegmentation]] = None
    iscrowd: int = Field(0, ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def bbox_dimensions_non_negative(cls, v: list[float]) -> list[float]:
        _, _, w, h = v
        if w < 0 or h < 0:
            raise ValueError(f"bbox width and height must be >= 0, got w={w}, h={h}")
        return v


class COCOCategory(BaseModel):
    """A category (class label) entry."""

    id: int
    name: str


class COCODataset(BaseModel):
    """Top-level COCO annotation file schema."""

    images: list[COCOImage]
    annotations: list[COCOAnnotation]
    categories: list[COCOCategory]

    @model_validator(mode="after")
    def image_ids_unique(self) -> COCODataset:
        ids = [img.id for img in self.images]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate image ids found")
        return self

    @model_validator(mode="after")
    def annotation_ids_unique(self) -> COCODataset:
        ids = [ann.id for ann in self.annotations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate annotation ids found")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> COCODataset:
        import json

        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)
