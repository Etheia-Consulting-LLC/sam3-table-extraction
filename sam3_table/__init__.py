try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency at import time
    def load_dotenv() -> bool:
        return False

from typing import Any

load_dotenv()

__all__ = [
    "TableDetection",
    "infer_table_bboxes",
    "infer_table_detections",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from sam3_table.inference import (
            TableDetection,
            infer_table_bboxes,
            infer_table_detections,
        )

        exports = {
            "TableDetection": TableDetection,
            "infer_table_bboxes": infer_table_bboxes,
            "infer_table_detections": infer_table_detections,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
