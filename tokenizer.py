"""A small corpus-built word tokenizer. No pretrained tokenizer is used."""
from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Iterable

# Words (including simple contractions), numbers, and individual punctuation.
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


class Tokenizer:
    def __init__(self, tokens: list[str]):
        if tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("Tokenizer vocabulary must begin with the four special tokens.")
        if len(set(tokens)) != len(tokens):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens.")
        self.tokens = tokens
        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

    @classmethod
    def build(cls, texts: Iterable[str], min_frequency: int = 1, max_vocab: int = 600) -> "Tokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(token.lower() for token in TOKEN_PATTERN.findall(text))
        words = [
            token for token, count in counts.most_common()
            if count >= min_frequency and token not in SPECIAL_TOKENS
        ]
        words = words[: max(0, max_vocab - len(SPECIAL_TOKENS))]
        return cls(SPECIAL_TOKENS + words)

    def tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        for value in ids:
            idx = int(value)
            if idx < 0 or idx >= len(self.tokens):
                token = "<UNK>"
            else:
                token = self.tokens[idx]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)

        # Simple English detokenization: punctuation closes the preceding word.
        result = ""
        closing = {".", ",", "!", "?", ";", ":", ")", "]", "}"}
        opening = {"(", "[", "{"}
        for piece in pieces:
            if not result:
                result = piece
            elif piece in closing:
                result += piece
            elif result.split()[-1] in opening:
                result += piece
            else:
                result += " " + piece
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": 1, "tokens": self.tokens}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != 1 or not isinstance(data.get("tokens"), list):
            raise ValueError("Unsupported or damaged tokenizer file.")
        return cls(data["tokens"])
