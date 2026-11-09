"""
Formal data schemas for loaded DataItems.

Important design rule:
- Schemas validate *loaded* data only
- They do NOT validate JSON metadata or filename references
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Protocol, runtime_checkable
import numpy as np


# ============================================================
# Base schema protocol
# ============================================================

@runtime_checkable
class DataSchema(Protocol):
    def validate(self, data) -> None:
        """
        Validate loaded data.
        Must raise TypeError / ValueError on failure.
        """
        ...


# ============================================================
# Scalar schema (float, int, etc.)
# ============================================================

@dataclass(frozen=True)
class ScalarSchema:
    dtype: type

    def validate(self, data) -> None:
        if not isinstance(data, self.dtype):
            raise TypeError(
                f"Expected scalar of type {self.dtype}, got {type(data)}"
            )


# ============================================================
# NumPy array schema
# ============================================================

@dataclass(frozen=True)
class ArraySchema:
    ndim: int
    shape: Optional[Tuple[Optional[int], ...]] = None
    dtype: Optional[type] = None

    def validate(self, data) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Expected np.ndarray, got {type(data)}"
            )

        if data.ndim != self.ndim:
            raise ValueError(
                f"Expected ndim={self.ndim}, got {data.ndim}"
            )

        if self.shape is not None:
            if len(self.shape) != data.ndim:
                raise ValueError(
                    f"Shape rank mismatch: expected {len(self.shape)}, got {data.ndim}"
                )

            for i, (expected, actual) in enumerate(zip(self.shape, data.shape)):
                if expected is not None and expected != actual:
                    raise ValueError(
                        f"Axis {i}: expected size {expected}, got {actual}"
                    )

        if self.dtype is not None:
            if not np.issubdtype(data.dtype, self.dtype):
                raise TypeError(
                    f"Expected dtype compatible with {self.dtype}, got {data.dtype}"
                )


# ============================================================
# Dictionary schema (structured data)
# ============================================================

@dataclass(frozen=True)
class DictSchema:
    """
    Schema for structured data represented as dictionaries.
    """
    fields: Dict[str, DataSchema]

    def validate(self, data) -> None:
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict, got {type(data)}"
            )

        for key, schema in self.fields.items():
            if key not in data:
                raise KeyError(
                    f"Missing required key '{key}'"
                )
            schema.validate(data[key])


@dataclass(frozen=True)
class ListSchema:
    element_schema: DataSchema

    def validate(self, data) -> None:
        if not isinstance(data, list):
            raise TypeError(f"Expected list, got {type(data)}")
        for i, x in enumerate(data):
            try:
                self.element_schema.validate(x)
            except Exception as e:
                raise TypeError(f"ListSchema element {i} failed validation: {e}")
