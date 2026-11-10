"""A tiny offline agent layer that routes a few explicit local tools."""
from __future__ import annotations

import ast
import operator
import re
from pathlib import Path


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str):
    """Evaluate basic arithmetic without eval() or arbitrary Python execution."""
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large for this calculator.")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](visit(node.operand))
        raise ValueError("Use numbers, parentheses, and + - * / // % ** only.")

    result = visit(tree)
    if isinstance(result, complex) or (isinstance(result, float) and not (float("-inf") < result < float("inf"))):
        raise ValueError("That calculation did not produce a finite real number.")
    return result


class ByteAgent:
    """Offline tool router + BYTE's small next-token model.

    This is a deliberately limited agent, not a general reasoning system.
    """

    def __init__(self, model, tokenizer, root: Path):
        self.model = model
        self.tokenizer = tokenizer
        self.root = Path(root)
        self.memory_path = self.root / "data" / "memory.txt"

    def _remember(self, note: str) -> str:
        note = note.strip()
        if not note:
            return "Tell me what to remember. Example: remember my project is called BYTE."
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(note.replace("\n", " ") + "\n")
        return "Saved to BYTE's local memory. This note is stored, but it does not change the model's weights."

    def _recall(self, query: str = "") -> str:
        if not self.memory_path.exists():
            return "BYTE's local memory is empty. Try: remember <something>."
        lines = [line.strip() for line in self.memory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if query:
            terms = [part.lower() for part in re.findall(r"[\w']+", query) if len(part) > 1]
            lines = [line for line in lines if all(term in line.lower() for term in terms)] if terms else []
        if not lines:
            return "I couldn't find a matching note in local memory."
        return "BYTE local memory:\n" + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines[-20:]))

    def _generate(self, prompt: str) -> str:
        ids = self.tokenizer.encode(prompt)
        generated = self.model.generate(ids, self.tokenizer.bos_id, self.tokenizer.eos_id,
                                        max_new_tokens=35, temperature=0.85, top_k=12)
        return self.tokenizer.decode(generated)

    def respond(self, message: str) -> str:
        text = message.strip()
        low = text.lower()
        if not text:
            return "Type a message. Try /help for commands."
        if low in {"/help", "help", "what can you do", "what can you do?"}:
            return ("BYTE offline agent tools:\n"
                    "• remember <note> — save a local note\n"
                    "• recall — list saved notes\n"
                    "• search memory for <words> — search saved notes\n"
                    "• calculate <expression> — basic arithmetic\n"
                    "• status — show model details\n"
                    "• generate <prompt> — sample text from the tiny model\n"
                    "Anything else is treated as a text-continuation experiment, not reliable Q&A.")
        if low == "/status" or low == "status":
            memory_count = 0
            if self.memory_path.exists():
                memory_count = sum(1 for line in self.memory_path.read_text(encoding="utf-8").splitlines() if line.strip())
            return (f"BYTE local status\nVocabulary: {len(self.tokenizer.tokens)} tokens\n"
                    f"Context window: {self.model.context_size} tokens\n"
                    f"Model: {self.model.embedding_size}-dim embeddings, {self.model.hidden_size}-unit hidden layer\n"
                    f"Saved notes: {memory_count}\nNetwork: offline; no API or internet connection required.")
        match = re.match(r"^(?:/remember|remember|save(?: this)?)(?:\s*[:,-]?\s+)(.+)$", text, flags=re.I | re.S)
        if match:
            return self._remember(match.group(1))
        if low in {"/recall", "recall", "what do you remember", "what do you remember?", "show my memory"}:
            return self._recall()
        match = re.match(r"^(?:search memory for|search my notes for)\s+(.+)$", text, flags=re.I)
        if match:
            return self._recall(match.group(1))
        match = re.match(r"^(?:/calculate|calculate|what is)\s+(.+?)[?]?$", text, flags=re.I)
        if match and re.fullmatch(r"[0-9\s()+\-*/%.]+", match.group(1)):
            try:
                return f"{match.group(1).strip()} = {safe_calculate(match.group(1))}"
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
                return f"Calculator error: {exc}"
        if low.startswith("/generate ") or low.startswith("generate "):
            prompt = text.split(" ", 1)[1]
            return "Experimental continuation (not a verified answer):\n" + self._generate(prompt)
        return ("I can route local tools, but my tiny experimental model is not capable of reliable general Q&A.\n\n"
                "Generated continuation (may be incoherent):\n" + self._generate(text))
