"""Abstract tokenizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    name: str

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        ...

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        ...

    @property
    @abstractmethod
    def eos_id(self) -> int:
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @abstractmethod
    def is_direct(self, word: str) -> bool:
        """Word maps to a single token without UNK or subword split."""
        ...

    def encode_line(self, line: str) -> list[int]:
        ids = self.encode(line)
        if not ids or ids[-1] != self.eos_id:
            ids = ids + [self.eos_id]
        return ids
