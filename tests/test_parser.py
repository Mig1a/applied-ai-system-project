"""
Unit tests for utils/parser.py
Run with: python -m pytest tests/ -v
"""

import unittest
import sys
import os
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.parser import chunk_text, extract_text_from_docx


class TestChunkText(unittest.TestCase):

    def test_short_text_returns_single_chunk(self):
        text = "This is a short resume with only a few words."
        chunks = chunk_text(text, chunk_size=400)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_long_text_returns_multiple_chunks(self):
        # 500 words
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        self.assertGreater(len(chunks), 1)

    def test_chunks_are_non_empty(self):
        text = " ".join(["word"] * 200)
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_empty_text_returns_empty_list(self):
        chunks = chunk_text("")
        self.assertEqual(chunks, [])

    def test_overlap_means_words_are_shared(self):
        words = [f"word{i}" for i in range(20)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        # The last few words of chunk 0 should appear at the start of chunk 1
        last_words_chunk0 = chunks[0].split()[-3:]
        first_words_chunk1 = chunks[1].split()[:3]
        self.assertEqual(last_words_chunk0, first_words_chunk1)

    def test_chunk_size_respected(self):
        text = " ".join(["word"] * 300)
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 100)

    def test_whitespace_only_text_returns_empty(self):
        chunks = chunk_text("     ")
        self.assertEqual(chunks, [])


class TestExtractTextFromDocx(unittest.TestCase):

    def test_invalid_bytes_raises_value_error(self):
        with self.assertRaises(ValueError):
            extract_text_from_docx(b"not a real docx file")

    def test_empty_bytes_raises_value_error(self):
        with self.assertRaises(ValueError):
            extract_text_from_docx(b"")


class TestResumeTextPipeline(unittest.TestCase):
    """Integration-style tests for the full text → chunk pipeline."""

    def test_chunk_then_join_preserves_content(self):
        original = " ".join([f"token{i}" for i in range(100)])
        chunks = chunk_text(original, chunk_size=30, overlap=5)
        joined = " ".join(chunks)
        # Every original token should appear somewhere in the joined output
        for token in original.split():
            self.assertIn(token, joined)

    def test_single_word_text(self):
        chunks = chunk_text("Python")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Python")

    def test_chunk_count_scales_with_text_length(self):
        short = " ".join(["w"] * 50)
        long = " ".join(["w"] * 500)
        chunks_short = chunk_text(short, chunk_size=100, overlap=10)
        chunks_long = chunk_text(long, chunk_size=100, overlap=10)
        self.assertGreater(len(chunks_long), len(chunks_short))


if __name__ == "__main__":
    unittest.main(verbosity=2)
