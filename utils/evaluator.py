"""
Evaluation / scoring module.

Provides keyword matching, skill gap detection, relevance scoring,
readability analysis, and a basic hallucination check.
"""

import re
import logging
from typing import Dict, List

import textstat

logger = logging.getLogger(__name__)

# Curated list of skills to look for across both resume and JD
TECH_SKILLS: List[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r",
    # Frontend
    "react", "vue", "angular", "html", "css", "next.js", "tailwind",
    # Backend / infra
    "node", "django", "flask", "fastapi", "spring", "express",
    "docker", "kubernetes", "terraform", "ansible", "linux",
    "aws", "azure", "gcp", "ci/cd", "devops", "microservices", "rest", "graphql",
    # Data / ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "spark",
    "tableau", "power bi", "sql", "postgresql", "mysql", "mongodb",
    "redis", "kafka", "hadoop", "airflow",
    # Version control / methodology
    "git", "agile", "scrum", "jira", "kanban",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "mentoring", "stakeholder management",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _word_set(text: str) -> set:
    return {w for w in _normalise(text).split() if len(w) > 2}


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def keyword_match_score(resume_text: str, job_description: str) -> Dict:
    """
    Measure how many of the JD's keywords also appear in the resume.
    Returns a dict with the numeric score (0-100), matched keywords, etc.
    """
    resume_words = _word_set(resume_text)
    jd_words = _word_set(job_description)

    if not jd_words:
        return {
            "score": 0.0,
            "matched_keywords": [],
            "matched_count": 0,
            "total_jd_keywords": 0,
        }

    matched = sorted(resume_words & jd_words)
    score = round(len(matched) / len(jd_words) * 100, 1)

    return {
        "score": score,
        "matched_keywords": matched[:30],
        "matched_count": len(matched),
        "total_jd_keywords": len(jd_words),
    }


def _skill_present(skill: str, text: str) -> bool:
    """Return True if *skill* appears as a whole word (or phrase) in *text*."""
    pattern = r"\b" + re.escape(skill) + r"\b"
    return bool(re.search(pattern, text))


def detect_missing_skills(resume_text: str, job_description: str) -> Dict:
    """
    Check TECH_SKILLS list against both texts to identify what the JD asks for
    that the candidate's resume does not mention.
    Uses whole-word matching to avoid false positives (e.g. 'r' inside 'great').
    """
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    present: List[str] = []
    missing: List[str] = []

    for skill in TECH_SKILLS:
        if _skill_present(skill, jd_lower):
            (present if _skill_present(skill, resume_lower) else missing).append(skill)

    return {
        "present_skills": present,
        "missing_skills": missing,
        "present_count": len(present),
        "missing_count": len(missing),
    }


def relevance_score(resume_text: str, job_description: str) -> float:
    """
    Jaccard-based overlap between resume and JD word sets, scaled to 0-100.
    """
    rw = _word_set(resume_text)
    jw = _word_set(job_description)
    if not rw or not jw:
        return 0.0
    union = rw | jw
    intersection = rw & jw
    # Scale: raw Jaccard is usually small; multiply to get a friendlier range
    raw = len(intersection) / len(union)
    return min(round(raw * 350, 1), 100.0)


def readability_score(text: str) -> Dict:
    """
    Compute Flesch Reading Ease and Flesch-Kincaid Grade Level using textstat.
    """
    try:
        flesch = textstat.flesch_reading_ease(text)
        grade = textstat.flesch_kincaid_grade(text)

        if flesch >= 70:
            label = "Easy to read"
        elif flesch >= 50:
            label = "Moderately easy"
        elif flesch >= 30:
            label = "Difficult"
        else:
            label = "Very difficult"

        return {
            "flesch_score": round(flesch, 1),
            "grade_level": round(grade, 1),
            "readability": label,
            "word_count": len(text.split()),
            "sentence_count": textstat.sentence_count(text),
        }
    except Exception as exc:
        logger.warning("Readability calculation failed: %s", exc)
        return {
            "flesch_score": 0.0,
            "grade_level": 0.0,
            "readability": "Unavailable",
            "word_count": len(text.split()),
            "sentence_count": 0,
        }


def hallucination_check(generated_text: str, source_text: str) -> Dict:
    """
    Heuristic: count words in the generated text that do NOT appear in the
    combined source (resume + JD).  A high ratio suggests potential
    hallucination.
    """
    STOP_WORDS = {
        "the", "and", "for", "are", "was", "were", "has", "have", "will",
        "with", "from", "this", "that", "you", "your", "our", "their", "its",
        "can", "may", "should", "would", "could", "been", "being", "more",
        "also", "well", "not", "all", "any", "one", "who", "how", "what",
        "when", "where", "why", "but", "they", "them", "these", "those",
        "into", "over", "than", "just", "such", "each", "both", "about",
        "out", "his", "her", "him", "she", "he", "we", "my", "an", "a",
    }

    gen_words = _word_set(generated_text) - STOP_WORDS
    source_words = _word_set(source_text)
    novel = gen_words - source_words

    if not gen_words:
        return {"grounding_score": 100.0, "novel_words_count": 0, "status": "Well grounded"}

    ratio = len(novel) / len(gen_words)
    grounding = round(max(0.0, (1 - ratio) * 100), 1)

    return {
        "grounding_score": grounding,
        "novel_words_count": len(novel),
        "status": "Well grounded" if grounding >= 65 else "Review recommended",
        "warning": grounding < 65,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_evaluation(
    resume_text: str,
    job_description: str,
    generated_outputs: Dict,
) -> Dict:
    """
    Run all evaluations and return a unified results dict.

    Args:
        resume_text: Raw resume string.
        job_description: Raw JD string.
        generated_outputs: Keys are output names (e.g. 'cover_letter'),
                           values are the generated text strings.
    """
    results: Dict = {}

    results["keyword_match"] = keyword_match_score(resume_text, job_description)
    results["skill_analysis"] = detect_missing_skills(resume_text, job_description)
    results["relevance_score"] = relevance_score(resume_text, job_description)

    # Readability per output
    results["readability"] = {}
    for key, text in generated_outputs.items():
        if isinstance(text, str) and text.strip():
            results["readability"][key] = readability_score(text)

    # Hallucination check on cover letter (most prone to drifting)
    cover_letter = generated_outputs.get("cover_letter", "")
    if cover_letter:
        source = resume_text + " " + job_description
        results["hallucination_check"] = hallucination_check(cover_letter, source)

    return results
