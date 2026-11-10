"""Dataset construction and training for BYTE's toy neural language model."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .model import TinyLanguageModel
from .tokenizer import Tokenizer


def load_texts(root: Path) -> list[str]:
    texts = []
    for file_path in (root / "data" / "corpus.txt", root / "data" / "memory.txt"):
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                texts.extend(line.strip() for line in content.splitlines() if line.strip())
    if not texts:
        raise RuntimeError("No training text found. Add lines to data/corpus.txt first.")
    return texts


def make_dataset(texts: list[str], tokenizer: Tokenizer, context_size: int):
    contexts: list[list[int]] = []
    targets: list[int] = []
    for text in texts:
        sequence = tokenizer.encode(text) + [tokenizer.eos_id]
        history = [tokenizer.bos_id] * context_size
        for target in sequence:
            contexts.append(history[-context_size:])
            targets.append(target)
            history.append(target)
    if not contexts:
        raise RuntimeError("Training corpus produced no tokens.")
    return np.asarray(contexts, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def train(root: Path, epochs: int = 60, batch_size: int = 64, learning_rate: float = 0.01,
          context_size: int = 4, seed: int = 7) -> list[float]:
    texts = load_texts(root)
    tokenizer = Tokenizer.build(texts, max_vocab=600)
    x, y = make_dataset(texts, tokenizer, context_size)
    model = TinyLanguageModel(len(tokenizer.tokens), context_size=context_size, seed=seed)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(order), batch_size):
            batch_ids = order[start:start + batch_size]
            losses.append(model.train_batch(x[batch_ids], y[batch_ids], learning_rate))
        average_loss = float(np.mean(losses))
        history.append(average_loss)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"epoch {epoch:>3}/{epochs} | loss {average_loss:.4f}")
    tokenizer.save(root / "models" / "tokenizer.json")
    model.save(root / "models" / "byte_model.npz")
    (root / "models" / "training_info.txt").write_text(
        f"texts={len(texts)}\ntokens={len(y)}\nvocabulary={len(tokenizer.tokens)}\n"
        f"context_size={context_size}\nepochs={epochs}\nfinal_loss={history[-1]:.6f}\n",
        encoding="utf-8",
    )
    return history
