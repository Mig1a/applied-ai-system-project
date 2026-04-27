"""
Unit tests for utils/evaluator.py
Run with: python -m pytest tests/ -v
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.evaluator import (
    keyword_match_score,
    detect_missing_skills,
    relevance_score,
    readability_score,
    hallucination_check,
)


class TestKeywordMatchScore(unittest.TestCase):

    def test_perfect_match(self):
        resume = "python developer with django and postgresql experience"
        jd = "python django postgresql"
        result = keyword_match_score(resume, jd)
        self.assertEqual(result["score"], 100.0)

    def test_zero_match(self):
        resume = "experienced chef with culinary arts background"
        jd = "python machine learning tensorflow aws"
        result = keyword_match_score(resume, jd)
        self.assertEqual(result["score"], 0.0)

    def test_partial_match(self):
        resume = "python developer with sql skills"
        jd = "python sql java kubernetes docker"
        result = keyword_match_score(resume, jd)
        self.assertGreater(result["score"], 0)
        self.assertLess(result["score"], 100)

    def test_empty_jd_returns_zero(self):
        result = keyword_match_score("some resume text", "")
        self.assertEqual(result["score"], 0)

    def test_returns_matched_keywords_list(self):
        resume = "python developer sql"
        jd = "python sql"
        result = keyword_match_score(resume, jd)
        self.assertIn("matched_keywords", result)
        self.assertIsInstance(result["matched_keywords"], list)

    def test_score_is_between_0_and_100(self):
        resume = "software engineer with python java and react experience"
        jd = "looking for python react node aws engineer"
        result = keyword_match_score(resume, jd)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)


class TestDetectMissingSkills(unittest.TestCase):

    def test_detects_missing_docker(self):
        resume = "python developer, knows sql and git"
        jd = "requires python, sql, git, and docker experience"
        result = detect_missing_skills(resume, jd)
        self.assertIn("docker", result["missing_skills"])

    def test_present_skills_detected(self):
        resume = "python developer with sql and git"
        jd = "python sql git docker"
        result = detect_missing_skills(resume, jd)
        self.assertIn("python", result["present_skills"])
        self.assertIn("sql", result["present_skills"])

    def test_counts_are_correct(self):
        resume = "python sql"
        jd = "python sql docker kubernetes"
        result = detect_missing_skills(resume, jd)
        self.assertEqual(result["present_count"], len(result["present_skills"]))
        self.assertEqual(result["missing_count"], len(result["missing_skills"]))

    def test_empty_resume_all_missing(self):
        resume = ""
        jd = "python docker kubernetes"
        result = detect_missing_skills(resume, jd)
        self.assertEqual(result["present_count"], 0)

    def test_no_skills_in_jd(self):
        resume = "python developer"
        jd = "great attitude and team player"
        result = detect_missing_skills(resume, jd)
        self.assertEqual(result["present_count"], 0)
        self.assertEqual(result["missing_count"], 0)


class TestRelevanceScore(unittest.TestCase):

    def test_identical_texts_score_high(self):
        text = "python developer with machine learning and sql experience"
        score = relevance_score(text, text)
        self.assertGreater(score, 50)

    def test_completely_different_texts_score_low(self):
        resume = "chef cook culinary arts restaurant"
        jd = "python kubernetes aws terraform devops"
        score = relevance_score(resume, jd)
        self.assertLess(score, 20)

    def test_score_is_between_0_and_100(self):
        score = relevance_score("some text here", "other text there")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_empty_inputs_return_zero(self):
        self.assertEqual(relevance_score("", ""), 0.0)
        self.assertEqual(relevance_score("text", ""), 0.0)


class TestReadabilityScore(unittest.TestCase):

    def test_returns_expected_keys(self):
        text = "This is a simple sentence. It is easy to read."
        result = readability_score(text)
        for key in ["flesch_score", "grade_level", "readability", "word_count", "sentence_count"]:
            self.assertIn(key, result)

    def test_word_count_is_accurate(self):
        text = "one two three four five"
        result = readability_score(text)
        self.assertEqual(result["word_count"], 5)

    def test_empty_text_does_not_crash(self):
        result = readability_score("")
        self.assertIn("word_count", result)

    def test_readability_label_is_string(self):
        text = "The quick brown fox jumps over the lazy dog near the river bank."
        result = readability_score(text)
        self.assertIsInstance(result["readability"], str)


class TestHallucinationCheck(unittest.TestCase):

    def test_fully_grounded_text_scores_high(self):
        source = "python developer with django and sql skills at company"
        generated = "python developer with django and sql skills"
        result = hallucination_check(generated, source)
        self.assertGreater(result["grounding_score"], 80)

    def test_completely_novel_text_scores_low(self):
        source = "python developer"
        generated = "neurosurgeon with twenty years specialising cardiothoracic procedures"
        result = hallucination_check(generated, source)
        self.assertLess(result["grounding_score"], 50)

    def test_returns_required_keys(self):
        result = hallucination_check("some text", "source text")
        for key in ["grounding_score", "novel_words_count", "status"]:
            self.assertIn(key, result)

    def test_grounding_score_between_0_and_100(self):
        result = hallucination_check("random words here", "other words there")
        self.assertGreaterEqual(result["grounding_score"], 0)
        self.assertLessEqual(result["grounding_score"], 100)

    def test_empty_generated_returns_100(self):
        result = hallucination_check("", "some source text")
        self.assertEqual(result["grounding_score"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
