import tempfile
import unittest
from pathlib import Path

from byte_core.agent import ByteAgent, safe_calculate
from byte_core.tokenizer import Tokenizer


class AgentTests(unittest.TestCase):
    def test_safe_calculator(self):
        self.assertEqual(safe_calculate("18 * (4 + 2)"), 108)
        self.assertEqual(safe_calculate("7 / 2"), 3.5)
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('whoami')")

    def test_memory_tools(self):
        class FakeModel:
            context_size = 4
            embedding_size = 16
            hidden_size = 48
            def generate(self, *args, **kwargs):
                return [1]
        tokenizer = Tokenizer.build(["hello BYTE"])
        with tempfile.TemporaryDirectory() as tmp:
            agent = ByteAgent(FakeModel(), tokenizer, Path(tmp))
            self.assertIn("Saved", agent.respond("remember BYTE is local"))
            self.assertIn("BYTE is local", agent.respond("recall"))
            self.assertIn("BYTE is local", agent.respond("search memory for BYTE"))


if __name__ == "__main__":
    unittest.main()
