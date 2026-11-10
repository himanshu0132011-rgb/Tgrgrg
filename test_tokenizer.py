import unittest
from byte_core.tokenizer import Tokenizer

class TokenizerTests(unittest.TestCase):
    def test_special_tokens_and_round_trip(self):
        tokenizer = Tokenizer.build(["Hello, BYTE! Hello."], max_vocab=30)
        ids = tokenizer.encode("Hello, BYTE!")
        self.assertTrue(ids)
        self.assertEqual(tokenizer.decode(ids), "hello, byte!")

    def test_unknown_token(self):
        tokenizer = Tokenizer.build(["one two three"])
        self.assertEqual(tokenizer.encode("zebra"), [tokenizer.unk_id])

if __name__ == "__main__":
    unittest.main()
