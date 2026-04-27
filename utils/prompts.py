"""
Prompt templates for every AI generation task.

All functions return a ready-to-send user message string.
Use get_system_prompt() for the shared system role message.
"""

from typing import Optional


def get_system_prompt() -> str:
    return (
        "You are an expert AI career assistant specialising in resume optimisation, "
        "cover letter writing, and interview coaching. "
        "You write with precision and professionalism. "
        "Always base responses strictly on the provided candidate context and job description — "
        "never invent credentials, companies, or metrics the candidate did not mention."
    )


def get_resume_bullets_prompt(context: str, job_description: str) -> str:
    return f"""You are an expert ATS resume writer. Using the candidate's background and the job description below, write 7 powerful, ATS-optimised resume bullet points.

CANDIDATE CONTEXT (from resume):
{context}

JOB DESCRIPTION:
{job_description}

RULES:
- Start every bullet with a strong past-tense action verb (Engineered, Reduced, Led, Delivered, Automated, etc.)
- Include at least one quantifiable result per bullet (%, $, time saved, scale)
- Naturally weave in keywords from the job description
- Follow the CAR pattern: Context → Action → Result
- Maximum two lines per bullet
- Do NOT reuse the same verb twice

Return ONLY the bullet points, one per line, each prefixed with "•".
No preamble, no headers, no explanation.
"""


def get_cover_letter_prompt(
    context: str,
    job_description: str,
    draft: Optional[str] = None,
) -> str:
    draft_block = (
        f"\nCANDIDATE'S DRAFT (refine and improve this):\n{draft}\n"
        if draft
        else ""
    )

    return f"""You are an expert cover letter writer. Write a compelling, personalised cover letter for this candidate.

CANDIDATE CONTEXT (from resume):
{context}

JOB DESCRIPTION:
{job_description}
{draft_block}

STRUCTURE:
1. Opening paragraph — specific role name, genuine enthusiasm, one headline achievement
2. Body paragraph 1 — most relevant technical experience mapped to JD requirements
3. Body paragraph 2 — soft skills, culture fit, why this company specifically
4. Closing paragraph — confident call to action, thank the reader

RULES:
- Professional yet warm tone
- Mirror key phrases from the JD naturally — do not stuff
- Do NOT use placeholder text like [Company Name]; write "your company" or "this team" instead
- Keep to 4 paragraphs, ~300 words total
- No bullet points inside the letter

Return the complete cover letter only. No commentary.
"""


def get_interview_questions_prompt(job_description: str, context: str) -> str:
    return f"""You are a senior interview coach. Generate 10 realistic, role-specific interview questions.

JOB DESCRIPTION:
{job_description}

CANDIDATE CONTEXT:
{context}

Generate exactly 10 questions in this distribution:
- 3 BEHAVIORAL questions starting with "Tell me about a time…" or "Describe a situation…"
- 4 TECHNICAL questions directly testing skills listed in the JD
- 2 SITUATIONAL questions starting with "What would you do if…" or "How would you handle…"
- 1 CULTURE FIT question about working style or values

Format each question like:
[BEHAVIORAL] <question>
[TECHNICAL] <question>
[SITUATIONAL] <question>
[CULTURE] <question>

No numbering, no preamble. Questions only.
"""


def get_star_answers_prompt(context: str, questions: str) -> str:
    return f"""You are an expert interview coach. Write concise STAR-method answers for each question below.

CANDIDATE BACKGROUND:
{context}

QUESTIONS:
{questions}

For EACH question write:
**Question:** <repeat the question>
**Situation:** <1-2 sentences of context>
**Task:** <what needed to be done>
**Action:** <specific steps taken — 2-3 sentences, most important part>
**Result:** <measurable outcome>

Each answer: 150-200 words. Natural, confident, first-person voice.
Separate answers with a blank line.
"""


def get_skill_gap_prompt(resume_text: str, job_description: str) -> str:
    return f"""You are a career development expert. Produce a structured skill gap analysis.

CANDIDATE RESUME:
{resume_text[:3_000]}

JOB DESCRIPTION:
{job_description[:2_000]}

Return your analysis in this EXACT structure:

## ✅ Matched Skills
List the top 8 skills the candidate has that the job requires.

## ❌ Critical Gaps
List up to 6 must-have skills the JD requires that the resume does NOT show.

## ⚠️ Nice-to-Have Gaps
List up to 4 preferred skills that are missing.

## 📚 Learning Roadmap
For each Critical Gap, provide:
- **Skill:** <name>
- **Resource:** <specific course name on Coursera / Udemy / official docs>
- **Time to acquire:** <X weeks>
- **Quick win:** <one concrete action to take this week>

## 📊 Overall Assessment
2-3 sentences on overall fit and the single most important thing to address.

Be specific and actionable. No filler.
"""
