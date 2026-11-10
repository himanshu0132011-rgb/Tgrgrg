#!/usr/bin/env python3
"""BYTE v0.1 — a tiny, local, from-scratch language-model experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from byte_core.model import TinyLanguageModel
from byte_core.agent import ByteAgent
from byte_core.tokenizer import Tokenizer
from byte_core.training import train

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "byte_model.npz"
TOKENIZER_PATH = ROOT / "models" / "tokenizer.json"
MEMORY_PATH = ROOT / "data" / "memory.txt"


def load_engine():
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        raise RuntimeError("BYTE has not been trained yet. Run: python byte.py train")
    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    model = TinyLanguageModel.load(MODEL_PATH)
    if model.vocab_size != len(tokenizer.tokens):
        raise RuntimeError("Model/tokenizer vocabulary mismatch. Run training again.")
    return model, tokenizer


def cmd_train(args):
    print("BYTE: training a tiny neural language model from random initial weights.")
    print("No pretrained model, API, or remote service is being used.")
    train(ROOT, epochs=args.epochs, batch_size=args.batch_size,
          learning_rate=args.learning_rate, context_size=args.context)
    print("Saved models/byte_model.npz and models/tokenizer.json")
    print("Training complete. This is a toy next-token predictor, not a capable chatbot.")


def cmd_generate(args):
    model, tokenizer = load_engine()
    prompt_ids = tokenizer.encode(args.prompt)
    generated = model.generate(prompt_ids, tokenizer.bos_id, tokenizer.eos_id,
                               max_new_tokens=args.tokens, temperature=args.temperature,
                               top_k=args.top_k, seed=args.seed)
    print(tokenizer.decode(generated))


def cmd_chat(_args):
    model, tokenizer = load_engine()
    agent = ByteAgent(model, tokenizer, ROOT)
    print("BYTE v0.2 — offline tool-using experimental agent. Type /help or /quit.")
    print("The agent can save/search notes and calculate; its tiny language model is not reliable general Q&A.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBYTE session ended.")
            return
        if prompt.lower() in {"/quit", "/exit"}:
            return
        if not prompt:
            continue
        print("byte> " + agent.respond(prompt))


def cmd_remember(args):
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_PATH.open("a", encoding="utf-8") as file:
        file.write(args.text.strip() + "\n")
    print("Saved locally to data/memory.txt.")
    print("BYTE will not learn this until you run: python byte.py train")


def cmd_status(_args):
    print("BYTE v0.2 status")
    print(f"Project folder: {ROOT}")
    print(f"Training corpus: {(ROOT / 'data' / 'corpus.txt').exists()}")
    print(f"Saved model: {MODEL_PATH.exists()}")
    print(f"Saved tokenizer: {TOKENIZER_PATH.exists()}")
    if TOKENIZER_PATH.exists():
        tok = Tokenizer.load(TOKENIZER_PATH)
        print(f"Vocabulary size: {len(tok.tokens)}")
    if (ROOT / "models" / "training_info.txt").exists():
        print("\n" + (ROOT / "models" / "training_info.txt").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="BYTE: a tiny from-scratch local AI experiment with an offline agent layer")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="build tokenizer and train the tiny neural model")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--context", type=int, default=4)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="continue a text prompt")
    p.add_argument("prompt", nargs="?", default="BYTE is")
    p.add_argument("--tokens", type=int, default=35)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("chat", help="open a simple interactive text-continuation session")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("remember", help="save a local text note for the next training run")
    p.add_argument("text", help="text to save")
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("status", help="show local project and model status")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"BYTE error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
