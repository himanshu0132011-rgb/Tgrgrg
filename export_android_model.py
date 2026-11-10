#!/usr/bin/env python3
"""Export the desktop NumPy checkpoint into BYTE Android's compact binary asset."""
from __future__ import annotations
import json
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "byte_model.npz"
TOKENIZER = ROOT / "models" / "tokenizer.json"
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"


def main() -> None:
    if not MODEL.exists() or not TOKENIZER.exists():
        raise SystemExit("Train BYTE first: python byte.py train")
    ASSETS.mkdir(parents=True, exist_ok=True)
    with np.load(MODEL) as data:
        vocab = int(data["vocab_size"])
        context = int(data["context_size"])
        embedding = int(data["embedding_size"])
        hidden = int(data["hidden_size"])
        with (ASSETS / "byte_weights.bin").open("wb") as f:
            f.write(b"BYTE")
            f.write(struct.pack(">5i", 1, vocab, context, embedding, hidden))
            for key in ("E", "W1", "b1", "W2", "b2"):
                values = np.asarray(data[key], dtype=np.float32).reshape(-1)
                f.write(struct.pack(">" + str(values.size) + "f", *values.tolist()))
    (ASSETS / "tokenizer.json").write_bytes(TOKENIZER.read_bytes())
    print(f"Exported Android assets: vocab={vocab}, context={context}, embedding={embedding}, hidden={hidden}")


if __name__ == "__main__":
    main()
