# Model Card — AI Job Copilot

**Project:** AI Job Copilot
**Author:** Million Aboye
**Course:** GMU AI 101 — Project 4 | Spring 2026
**Version:** 1.0
**Last Updated:** April 2026

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Intended Use](#2-intended-use)
3. [System Architecture Summary](#3-system-architecture-summary)
4. [Testing Summary](#4-testing-summary)
5. [Human Evaluation](#5-human-evaluation)
6. [Reflection and Ethics](#6-reflection-and-ethics)
7. [Collaboration with AI](#7-collaboration-with-ai)

---

## 1. Model Overview

AI Job Copilot is a Retrieval-Augmented Generation (RAG) application built on top of OpenAI's GPT-4o-mini. It is not a standalone model — it is a system that wraps a foundation model with structured retrieval, prompt engineering, and a local evaluation pipeline to produce job application materials tailored to a specific candidate and role.

| Property | Value |
|---|---|
| **Base Model** | GPT-4o-mini (OpenAI) |
| **Embedding Model** | text-embedding-3-small (OpenAI) |
| **Vector Store** | FAISS (in-memory, cosine similarity) |
| **Frontend** | Streamlit |
| **Evaluation** | Local (textstat, set-based scoring) + AI self-scoring |
| **Input** | Resume (PDF/DOCX) + Job Description (text) |
| **Output** | Resume bullets, cover letter, interview questions, STAR answers, skill gap report |

---

## 2. Intended Use

### Primary Use Case
Helping individual job seekers tailor their application materials to a specific job posting faster and more effectively than doing it manually.

### Who It Is For
- Recent graduates applying for their first roles
- Professionals changing careers or industries
- Anyone applying to high volumes of jobs who needs to customise each application

### What It Is NOT For
- Fabricating experience or credentials a candidate does not have
- Mass-generating applications with zero personalisation
- Replacing professional career coaches or human judgment
- Use by recruiters or employers to evaluate or screen candidates

---

## 3. System Architecture Summary

```
Resume Upload → Document Parser → Text Chunker
                                        ↓
                               OpenAI Embeddings
                                        ↓
                               FAISS Vector Store
                                        ↓
Job Description ──────────────→ RAG Retrieval (top-6 chunks)
                                        ↓
                               GPT-4o-mini Generator
                              (5 separate prompt calls)
                                        ↓
                    ┌───────────────────┴───────────────────┐
                    ↓                                       ↓
             Local Evaluator                     AI Confidence Scorer
   (keyword match, relevance,              (model rates each output 1–10)
    readability, hallucination)
                    ↓                                       ↓
                    └───────────────────┬───────────────────┘
                                        ↓
                              Output Dashboard
                     (edit, rate, download, save feedback)
```

---

## 4. Testing Summary

### 4.1 Automated Unit Tests

Two test files cover the core logic modules with **36 tests across 6 test classes**, all passing.

**`tests/test_evaluator.py` — 24 tests**

| Test Class | What It Tests | Result |
|---|---|---|
| `TestKeywordMatchScore` | Perfect match, zero match, partial, empty JD, score bounds, return type | ✅ 6/6 |
| `TestDetectMissingSkills` | Missing skill detection, present skills, count accuracy, empty inputs | ✅ 5/5 |
| `TestRelevanceScore` | Identical texts, mismatched texts, empty inputs, score bounds | ✅ 4/4 |
| `TestReadabilityScore` | Return keys, word count accuracy, empty text crash-safety, label type | ✅ 4/4 |
| `TestHallucinationCheck` | Grounded text scores high, novel text scores low, return keys, bounds, empty input | ✅ 5/5 |

**`tests/test_parser.py` — 12 tests**

| Test Class | What It Tests | Result |
|---|---|---|
| `TestChunkText` | Short text, long text, non-empty chunks, empty input, overlap correctness, chunk size cap, whitespace | ✅ 7/7 |
| `TestExtractTextFromDocx` | Invalid bytes, empty bytes raise ValueError | ✅ 2/2 |
| `TestResumeTextPipeline` | Content preservation after chunking, single word, chunk count scaling | ✅ 3/3 |

**Run command:**
```bash
python -m pytest tests/ -v
```

**Final result: 36 passed, 0 failed, 1 deprecation warning (PyPDF2)**

---

### 4.2 Bug Discovered During Testing

Testing caught a real logic error. The `detect_missing_skills` function originally used Python's `in` operator for substring matching:

```python
# BUGGY: "r" in "great attitude and team player" → True (false positive)
if skill in jd_lower:
```

The skill `"r"` (R programming language) matched as a substring inside words like "great", "player", and "attitude", reporting them as present when they were not.

**Fix:** Replaced with whole-word regex matching using word boundaries:

```python
# FIXED: uses \b word boundary so "r" only matches the standalone word
def _skill_present(skill: str, text: str) -> bool:
    pattern = r"\b" + re.escape(skill) + r"\b"
    return bool(re.search(pattern, text))
```

This is a meaningful fix — without it, the skill gap analysis would silently undercount missing skills for any JD containing common English words.

---

### 4.3 Logging and Error Handling

Every step in the pipeline is logged to `logs/app.log` with timestamps. Sample log output from a real run:

```
2026-04-27 14:32:01 [INFO] __main__ — AI Job Copilot started.
2026-04-27 14:32:18 [INFO] __main__ — Pipeline started — resume length: 3842 chars, JD length: 1204 chars
2026-04-27 14:32:18 [INFO] __main__ — Chunked resume into 11 chunks
2026-04-27 14:32:21 [INFO] utils.rag — Embedding 11 chunks …
2026-04-27 14:32:23 [INFO] utils.rag — Vector store ready (11 vectors).
2026-04-27 14:32:23 [INFO] utils.rag — Retrieved 6 relevant chunks
2026-04-27 14:32:23 [INFO] __main__ — OpenAI call — prompt length: 2847 chars, max_tokens: 800
2026-04-27 14:32:26 [INFO] __main__ — OpenAI call succeeded — response length: 612 chars
2026-04-27 14:32:27 [INFO] __main__ — Confidence score for resume bullets: 8/10
...
2026-04-27 14:32:54 [INFO] __main__ — Pipeline complete. Confidence scores: {'resume_bullets': 8, 'cover_letter': 7, 'interview_questions': 9, 'star_answers': 6, 'skill_gap': 8}
```

Handled error classes: `AuthenticationError`, `RateLimitError`, `ValueError` (bad file), and a general `Exception` catch-all — each displays a specific, actionable message to the user.

---

### 4.4 AI Confidence Scoring

After generating each output, the system makes a lightweight follow-up call asking GPT-4o-mini to rate its own confidence from 1–10 based on how grounded the content was in the provided context.

**Observed pattern across test runs:**
- Interview questions consistently scored **8–9/10** — the model is confident because questions can be derived directly from the JD requirements
- STAR answers scored lower (**5–7/10**) — the model correctly recognised it was inferring specific anecdotes that the candidate would need to fill in with real details
- Cover letters scored **6–8/10** depending on how detailed the resume was

This self-scoring is a meaningful signal: a low confidence score on STAR answers tells the user "review these carefully and personalise them" — which is exactly the right guidance.

---

## 5. Human Evaluation

The app includes a built-in human feedback panel at the bottom of the results dashboard. Users rate each of the five outputs on a 1–5 star scale and optionally add free-text notes. Feedback is saved to `outputs/feedback.json` alongside the AI confidence scores and a timestamp for future review.

**Sample feedback record (`outputs/feedback.json`):**
```json
{
  "resume_bullets": { "stars": 4, "label": "⭐⭐⭐⭐" },
  "cover_letter":   { "stars": 3, "label": "⭐⭐⭐" },
  "interview_questions": { "stars": 4, "label": "⭐⭐⭐⭐" },
  "star_answers":   { "stars": 2, "label": "⭐⭐" },
  "skill_gap":      { "stars": 5, "label": "⭐⭐⭐⭐⭐" },
  "overall_notes": "STAR answers were too generic — they didn't reference my actual projects. Cover letter was good but used 'your company' which felt impersonal.",
  "timestamp": "2026-04-27T14:33:10",
  "confidence_scores": {
    "resume_bullets": 8,
    "cover_letter": 7,
    "interview_questions": 9,
    "star_answers": 6,
    "skill_gap": 8
  }
}
```

**Key finding from human evaluation:**

The AI's own confidence scores aligned well with human ratings. STAR answers received the lowest human rating (2 stars) and also the lowest AI confidence score (6/10). The model correctly identified its own weakest output. This suggests the confidence scoring mechanism has real diagnostic value — it is not just flattery.

The skill gap analysis consistently received the highest human ratings (4–5 stars). Users found it the most immediately actionable output because it named specific resources and gave a concrete learning timeline, not just a list of missing buzzwords.

---

## 6. Reflection and Ethics

### 6.1 Limitations and Biases in the System

**Language bias:** The system was built and tested entirely in English. Resumes or job descriptions in other languages will produce poor or nonsensical results. This excludes a large portion of the global job-seeking population.

**Keyword-based evaluation is shallow:** The keyword match score and skill gap detector use string matching, not semantic understanding. A resume that says "I manage distributed infrastructure at scale" scores zero for "kubernetes" even though the candidate may have deep relevant experience. The score is a useful proxy, not a ground truth.

**Skills taxonomy bias:** The 60-skill list in `evaluator.py` was manually curated and leans heavily toward software engineering and data roles. A graphic designer, nurse, or teacher using this tool would receive a skill gap analysis that is largely irrelevant to their field. The system does not adapt to the industry of the role.

**Metric bias:** The resume bullets are optimised for ATS keyword matching, which reflects how large-company hiring works. This may disadvantage candidates applying to small companies or creative fields where authentic, narrative resumes outperform keyword-dense ones.

**PDF parsing limitations:** The parser cannot extract text from scanned or image-only PDFs. Candidates whose resumes were created as images (common with professionally designed templates) will get no usable output. This disproportionately affects users who cannot afford or access Word-based templates.

**Thin resume bias:** The RAG pipeline retrieves up to 6 chunks from the resume. A candidate with a sparse resume (recent graduate, career change, employment gap) provides fewer signals for the model to work from, resulting in more generic outputs. The system inadvertently performs better for candidates who already have strong resumes — the people who need the least help.

---

### 6.2 Could This AI Be Misused?

**Yes — here are the realistic misuse scenarios:**

**Fabrication of credentials:** A user could paste a job description for a role they are completely unqualified for and ask the system to generate resume bullets. The model might produce plausible-sounding bullets that reference skills and experiences the candidate does not actually have, effectively helping someone lie on their application.

**Mitigation:** The system prompt explicitly instructs the model: *"never invent credentials, companies, or metrics the candidate did not mention."* The RAG pipeline grounds every generation call in the actual resume text. The hallucination check flags outputs that stray far from the source material. However, none of these are hard technical locks — a motivated user can still misuse the outputs.

**Spam applications:** The tool makes it fast to generate tailored applications. A user could apply to hundreds of jobs with minimal genuine interest, flooding recruiter inboxes and making it harder for serious candidates to stand out.

**Mitigation:** This is harder to prevent technically. The ethical framing in the app positions it as a tool for tailoring real applications, not mass generation. The human review step (inline editing, star ratings) creates friction that discourages purely automated use.

**Impersonation or proxy applications:** Someone could upload another person's resume and generate materials on their behalf without the resume owner's knowledge or consent.

**Mitigation:** Nothing in the current system prevents this. A future version with user authentication and explicit consent agreements would help. For now, this is flagged as an open ethical risk.

**The core principle:** This tool is designed to help candidates express their real qualifications more effectively — not to help them misrepresent themselves. The distinction matters, and users should understand it.

---

### 6.3 What Surprised Me While Testing

**The AI's confidence scores were self-aware in a useful way.** I expected the model to rate all of its outputs 8–9/10 as a kind of sycophantic default. Instead, it reliably gave STAR answers the lowest scores (5–7) and explained that it was making inferences beyond what the resume explicitly stated. That is accurate self-assessment. It surprised me because I had assumed self-evaluation would be unreliable — but for structured writing tasks with clear grounding criteria, the model has a reasonable sense of where it is guessing.

**The bug found by the tests was a real problem, not a toy case.** When I wrote the test `test_no_skills_in_jd`, I expected it to pass trivially. The fact that it failed — because `"r"` matched as a substring inside "great" and "player" — was a genuine logic error that would have corrupted skill gap results in production. This reminded me that testing is not just about confirming code works; it is about discovering the edge cases your intuition missed. I would not have caught this by reading the code. I only caught it by writing a test that forced the function to confront an edge case.

**Short resumes exposed a gap in the RAG design.** When I tested with a minimal resume (under 200 words), the vector store had only 2–3 chunks, which meant retrieval returned redundant content. The generated outputs became noticeably more generic. The system silently degrades rather than warning the user. I added a minimum-content advisory in the UI, but the deeper lesson is that RAG systems need to be tested at the edges of their input distribution, not just with ideal inputs.

---

## 7. Collaboration with AI

This project was built in collaboration with Claude (Anthropic), used as a coding assistant throughout development. The collaboration was iterative: I described what I needed, Claude wrote the code, I reviewed and tested it, and we debugged together when things broke. Below is an honest account of where that collaboration worked and where it did not.

---

### One Instance Where the AI Was Genuinely Helpful

**The RAG architecture decision.**

Early in the project, my instinct was to paste the full resume into every prompt. It worked — the outputs were usable. But Claude flagged that this approach was both wasteful and imprecise: the model receives a lot of irrelevant information for any given generation task, which dilutes the signal.

Claude proposed using FAISS to embed the resume into a vector store and query it with the job description as the search key. This way, for a DevOps role, the top retrieved chunks would surface the most DevOps-relevant experience — not the candidate's unrelated coursework or part-time retail job.

This was the most important architectural decision in the project, and I would not have made it on my own at this stage. The suggestion came with a clear explanation of *why* it was better (precision, token efficiency, cost), not just *what* to do. That reasoning helped me understand the principle, not just follow an instruction — which meant I could apply it to future projects independently.

---

### One Instance Where the AI's Suggestion Was Flawed

**The initial confidence scoring implementation.**

When I asked Claude to add AI confidence scoring, it suggested embedding a confidence rating inside each generation prompt — asking the model to append a score at the end of its own output, like:

```
• Built a data pipeline processing 2M records/day...
• Reduced API latency by 40%...
[CONFIDENCE: 8/10]
```

The idea was to parse that tag out of the response before displaying it to the user.

This was flawed for two reasons:

1. **It corrupted the output format.** The score tag appeared unpredictably — sometimes mid-paragraph, sometimes missing entirely, sometimes formatted as "Confidence: 8 out of 10" instead of the expected tag. Parsing it reliably would have required fragile regex that would break on variations.

2. **It conflated the generation task with the evaluation task.** Asking the model to generate a cover letter AND rate its own confidence in the same response creates competing instructions. The generation quality dropped slightly because part of the model's attention was on the meta-task.

The better approach — which I pushed back on and we agreed to implement instead — was a separate, minimal follow-up call dedicated solely to confidence scoring. This kept generation and evaluation cleanly separated, made the scoring more reliable (the model only had to output a single digit), and matched the architecture principle already established in the rest of the project: one focused call per task is better than one overloaded call.

**The lesson:** Claude's first suggestion prioritised simplicity (fewer API calls) over correctness. When I tested it and saw the format instability, I recognised the flaw and proposed the fix. The AI accepted the correction immediately and helped implement the cleaner version. Good collaboration means knowing when to push back.

---

*Model Card for AI Job Copilot v1.0 | GMU AI 101 — Project 4 | Spring 2026*
