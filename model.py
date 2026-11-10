"""A tiny context-window neural language model implemented with NumPy."""
from __future__ import annotations

from pathlib import Path
import numpy as np


class TinyLanguageModel:
    """A small MLP next-token predictor; it is NOT a transformer or an LLM."""

    def __init__(self, vocab_size: int, context_size: int = 4, embedding_size: int = 16,
                 hidden_size: int = 48, seed: int = 7):
        if vocab_size < 5:
            raise ValueError("Vocabulary is too small to train a language model.")
        self.vocab_size = int(vocab_size)
        self.context_size = int(context_size)
        self.embedding_size = int(embedding_size)
        self.hidden_size = int(hidden_size)
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.08, (vocab_size, embedding_size)).astype(np.float32)
        fan_in = context_size * embedding_size
        self.W1 = (rng.normal(0, np.sqrt(2 / fan_in), (fan_in, hidden_size))).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = (rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, vocab_size))).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)
        self._adam_m = [np.zeros_like(p) for p in self.parameters()]
        self._adam_v = [np.zeros_like(p) for p in self.parameters()]
        self._step = 0

    def parameters(self):
        return [self.E, self.W1, self.b1, self.W2, self.b2]

    def _forward(self, contexts: np.ndarray):
        embedded = self.E[contexts]
        flat = embedded.reshape(contexts.shape[0], -1)
        hidden = np.tanh(flat @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return flat, hidden, logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def train_batch(self, contexts: np.ndarray, targets: np.ndarray, learning_rate: float = 0.01) -> float:
        batch = contexts.shape[0]
        flat, hidden, logits = self._forward(contexts)
        probs = self._softmax(logits)
        loss = -np.log(np.maximum(probs[np.arange(batch), targets], 1e-9)).mean()

        dlogits = probs
        dlogits[np.arange(batch), targets] -= 1.0
        dlogits /= batch
        dW2 = hidden.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dhidden = dlogits @ self.W2.T
        dz = dhidden * (1.0 - hidden * hidden)
        dW1 = flat.T @ dz
        db1 = dz.sum(axis=0)
        dembedded = (dz @ self.W1.T).reshape(batch, self.context_size, self.embedding_size)
        dE = np.zeros_like(self.E)
        np.add.at(dE, contexts.reshape(-1), dembedded.reshape(-1, self.embedding_size))

        grads = [dE, dW1, db1, dW2, db2]
        self._step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i, (param, grad) in enumerate(zip(self.parameters(), grads)):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * grad
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * (grad * grad)
            m_hat = self._adam_m[i] / (1 - beta1 ** self._step)
            v_hat = self._adam_v[i] / (1 - beta2 ** self._step)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return float(loss)

    def next_logits(self, context: list[int]) -> np.ndarray:
        if not context:
            context = []
        context = context[-self.context_size:]
        # The first special token is used as left padding by the caller.
        if len(context) < self.context_size:
            context = [0] * (self.context_size - len(context)) + context
        x = np.asarray([context], dtype=np.int64)
        _, _, logits = self._forward(x)
        return logits[0]

    def generate(self, prompt_ids: list[int], bos_id: int, eos_id: int, max_new_tokens: int = 40,
                 temperature: float = 0.9, top_k: int = 12, seed: int | None = None) -> list[int]:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        rng = np.random.default_rng(seed)
        generated = list(prompt_ids)
        context = [bos_id] * self.context_size + generated
        for _ in range(max_new_tokens):
            logits = self.next_logits(context)
            logits = logits / temperature
            if top_k > 0 and top_k < self.vocab_size:
                keep = np.argpartition(logits, -top_k)[-top_k:]
                mask = np.ones(self.vocab_size, dtype=bool)
                mask[keep] = False
                logits[mask] = -np.inf
            logits -= np.max(logits)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            next_id = int(rng.choice(self.vocab_size, p=probabilities))
            if next_id == eos_id:
                break
            generated.append(next_id)
            context.append(next_id)
        return generated

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, E=self.E, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            vocab_size=self.vocab_size, context_size=self.context_size,
            embedding_size=self.embedding_size, hidden_size=self.hidden_size,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyLanguageModel":
        with np.load(path) as data:
            model = cls(int(data["vocab_size"]), int(data["context_size"]),
                        int(data["embedding_size"]), int(data["hidden_size"]))
            for name in ("E", "W1", "b1", "W2", "b2"):
                setattr(model, name, data[name].astype(np.float32))
        return model
