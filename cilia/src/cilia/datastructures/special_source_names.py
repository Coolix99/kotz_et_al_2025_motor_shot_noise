from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SpecialSourceName:
    key: str
    description: Optional[str] = None

ORIGINAL = SpecialSourceName(
    key="original",
    description="The 'most' original data what exists for that type of data",
)

GIVEN = SpecialSourceName(
    key="given",
    description="Given from the dataset but derived from other equantities",
)

GROUNDTHRUTH = SpecialSourceName(
    key="ground_thruth",
    description="From simulations if values can be assumed as exact",
)
