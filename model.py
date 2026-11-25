from typing import Any, Literal

from pydantic import BaseModel, Field

class Pronunciation(BaseModel):
    char: str
    pron: str


class GenshinTerm(BaseModel):
    id: str
    en: str
    ja: str

    zhCN: str | None = None
    zhTW: str | None = None

    pronunciationJa: str | None = None

    notes: str | None = None
    notesZh: str | None = None
    notesZhTW: str | None = None
    notesEn: str | None = None

    variants: dict[str, list[str]] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    pinyins: list[Pronunciation] = Field(default_factory=list)
    zhuyins: list[Pronunciation] = Field(default_factory=list)

    examples: list[dict[str, Any]] = Field(default_factory=list)


class RankedTerm(BaseModel):
    """A glossary match; semantic distance is ChromaDB's raw distance score."""

    rank: int = Field(ge=1)
    match_type: Literal["exact", "semantic"]
    chroma_distance: float | None = Field(
        default=None,
        description="Raw ChromaDB distance; lower values rank closer and are not probabilities.",
    )
    term: GenshinTerm
