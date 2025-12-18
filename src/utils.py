from dataclasses import dataclass
from typing import List


@dataclass
class Paper:
    arxiv_id: str
    title: str | None = None
    abstract: str | None = None
    authors: List[str] | None = None
