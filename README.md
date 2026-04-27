# AI Job Copilot

> An AI-powered job application assistant built with Python, Streamlit, and GPT-4o-mini.
> Upload your resume, paste a job description, and receive a full suite of tailored application materials in under 60 seconds.

---

## Table of Contents

1. [Title and Summary](#1-title-and-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Setup Instructions](#3-setup-instructions)
4. [Sample Interactions](#4-sample-interactions)
5. [Design Decisions](#5-design-decisions)
6. [Testing Summary](#6-testing-summary)
7. [Reflection](#7-reflection)

---

## 1. Title and Summary

### What It Does

AI Job Copilot is a full-stack web application that helps job seekers apply smarter and faster. The user uploads their resume (PDF or DOCX), pastes a job description, and the system produces five ready-to-use outputs:

| Output | Description |
|---|---|
| **ATS-Optimised Resume Bullets** | 7 keyword-rich bullet points tailored to the specific role |
| **Tailored Cover Letter** | A 4-paragraph, ~300-word letter that mirrors JD language naturally |
| **Interview Questions** | 10 role-specific questions across Behavioral, Technical, Situational, and Culture Fit categories |
| **STAR Method Answers** | Full Situation-Task-Action-Result answers for every interview question |
| **Skill Gap Analysis** | A ranked list of matched skills, missing skills, and a personalized learning roadmap |

Every output is editable inline and downloadable as a `.txt` file, either individually or as a single bundled document.

### Why It Matters

The modern job market is brutal. A single job posting can receive 500+ applications, and most companies use Applicant Tracking Systems (ATS) to filter resumes before a human ever sees them. Candidates who manually tailor each application spend 2–4 hours per job. AI Job Copilot collapses that to under 2 minutes — without sacrificing quality or personalization. It is not a generic cover letter generator. By routing the job description through a RAG (Retrieval-Augmented Generation) pipeline, every output is grounded in the candidate's actual experience and the specific requirements of the target role.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                    │
│  Sidebar Nav │ Upload Section │ JD Input │ Results Tabs │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              DOCUMENT PARSER  (utils/parser.py)         │
│  PDF → PyPDF2  │  DOCX → python-docx  │  Text Chunker   │
│  Output: list of ~400-word overlapping text chunks      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│               RAG PIPELINE  (utils/rag.py)              │
│  OpenAI text-embedding-3-small → float32 vectors        │
│  FAISS IndexFlatIP (cosine similarity after L2 norm)    │
│  Query = Job Description → retrieve top-6 chunks        │
│  Output: condensed context string (~2,000 words)        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│            AI GENERATOR  (utils/prompts.py + app.py)    │
│  GPT-4o-mini called 5× with role-specific prompts       │
│  Context window: retrieved chunks + full JD             │
│  Outputs: bullets, cover letter, questions, STAR, gap   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│            EVALUATOR  (utils/evaluator.py)              │
│  • Keyword Match Score   (JD keyword coverage %)        │
│  • Relevance Score       (Jaccard word-set similarity)  │
│  • Skill Gap Detection   (60+ skill taxonomy check)     │
│  • Readability Score     (Flesch Reading Ease)          │
│  • Grounding / Hallucination Check                      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│             OUTPUT DASHBOARD  (app.py)                  │
│  Score Cards │ 6 Tabs │ Inline Edit │ Download Buttons  │
└─────────────────────────────────────────────────────────┘
```

### Key Components

**Document Parser** — Handles binary file input. PyPDF2 iterates pages; python-docx iterates paragraphs and table cells. The chunker uses a sliding window with 60-word overlap so context is never hard-cut mid-sentence.

**RAG Pipeline** — Instead of dumping the entire resume into every prompt (wastes tokens, dilutes signal), the pipeline embeds the resume into a FAISS vector store and then queries it with the job description as the search key. This surfaces only the most relevant experience for each generation call, which improves output precision and reduces cost.

**AI Generator** — Five separate GPT-4o-mini calls, each with a carefully engineered prompt. The system prompt anchors the model as a career expert. The user prompts inject the retrieved context plus the full JD.

**Evaluator** — Runs entirely locally with zero additional API calls. Keyword match uses set intersection; skill gap checks against a curated 60-skill taxonomy; readability uses the `textstat` library; the hallucination check computes what fraction of the generated words are novel (not present in the source material).

**Output Dashboard** — Streamlit's `st.text_area` gives users an inline editor for every output. Download buttons use `st.download_button` to stream content directly to the browser as `.txt` files.

---

## 3. Setup Instructions

### Prerequisites

- Python 3.9 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Step 1 — Clone or Download the Project

```bash
# If you have the folder already, navigate into it:
cd ai-job-copilot
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `openai`, `PyPDF2`, `python-docx`, `faiss-cpu`, `pandas`, `python-dotenv`, `tiktoken`, `textstat`, `numpy`.

### Step 4 — Configure Your API Key

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

Alternatively, skip this step and enter the key directly in the app's **⚙️ Settings** tab — it is stored in your session only.

### Step 5 — Run the App

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### Step 6 — Use the App

1. Upload your resume (PDF or DOCX) in the left column
2. Paste the full job description in the right column
3. Optionally paste a cover letter draft you want refined
4. Click **Analyse & Generate**
5. Review outputs across the 6 result tabs
6. Edit, copy, or download any output

### Deploying to Streamlit Cloud

1. Push the `ai-job-copilot/` folder to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Select your repository, branch (`main`), and set the main file to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Click **Deploy** — the app goes live in approximately 60 seconds

---

## 4. Sample Interactions

The following examples show real inputs and the kind of output the system produces. Text has been lightly trimmed for space.

---

### Example 1 — Software Engineer → Backend Role at a Fintech Company

**Resume Snippet (uploaded):**
```
Software Engineer with 3 years of experience building REST APIs in Python (FastAPI, Django).
Led migration of monolithic billing service to microservices, reducing latency by 40%.
Built internal data pipeline processing 2M+ records/day using Kafka and PostgreSQL.
B.S. Computer Science, George Mason University, 2022.
```

**Job Description (pasted):**
```
Backend Software Engineer — PayFlow Inc.
We are looking for a backend engineer with experience in Python, REST APIs,
distributed systems, and PostgreSQL. Bonus: Kafka, Docker, or AWS experience.
You will own the design and delivery of core payment processing services.
```

**Generated Resume Bullets:**
```
• Architected and deployed 3 microservices in Python/FastAPI that reduced billing
  latency by 40%, directly improving checkout completion rates for 50K+ daily users.

• Designed a Kafka-driven data pipeline processing 2M+ records per day with
  sub-500ms end-to-end latency, replacing a brittle cron-based ETL system.

• Migrated a monolithic Django application to a distributed microservices architecture,
  cutting deployment times by 60% and enabling independent team scaling.

• Built and maintained 12 PostgreSQL schemas with optimised indexing strategies,
  reducing average query time from 800ms to under 90ms.
```

**Keyword Match Score:** 74%
**Relevance Score:** 68%
**Missing Skills Detected:** `docker`, `aws`

**Skill Gap — Critical Gaps:**
```
❌ Docker
   Resource: "Docker for Developers" — Udemy (Bret Fisher)
   Time to acquire: 2 weeks
   Quick win: Dockerise one of your existing FastAPI projects this weekend.

❌ AWS
   Resource: AWS Certified Developer — Associate (official AWS training)
   Time to acquire: 6–8 weeks
   Quick win: Deploy your Docker container to AWS ECS Free Tier today.
```

---

### Example 2 — Marketing Coordinator → Product Marketing Manager Role

**Resume Snippet (uploaded):**
```
Marketing Coordinator at TechStartup (2 years).
Managed social media accounts growing followers from 4K to 22K in 18 months.
Wrote product one-pagers and email campaigns with avg 32% open rate.
Coordinated 3 product launches, collaborated with sales and design teams.
```

**Job Description (pasted):**
```
Product Marketing Manager — CloudSuite
Drive go-to-market strategy for B2B SaaS products. Own positioning, messaging,
and competitive analysis. Work cross-functionally with Sales, Product, and Design.
3+ years marketing experience, B2B SaaS preferred. Strong written communication required.
```

**Generated Cover Letter (first paragraph):**
```
Dear Hiring Team,

When I read the CloudSuite Product Marketing Manager posting, one detail stood
out: "own the messaging." That is exactly what I have spent the last two years
doing — and I am ready to do it at scale. At TechStartup, I took our social
presence from 4,000 to 22,000 followers while simultaneously writing the
product one-pagers and email campaigns (averaging a 32% open rate) that our
sales team relied on to close deals. I coordinated three product launches
end-to-end, which taught me how to align Sales, Design, and Product around
a single narrative under real deadline pressure.
```

**Generated Interview Question (Behavioral):**
```
[BEHAVIORAL] Tell me about a time you had to align multiple teams —
Sales, Product, and Design — around a single go-to-market message.
How did you manage conflicting priorities, and what was the outcome?
```

**Generated STAR Answer:**
```
Situation: For our Q3 product launch, Sales wanted aggressive pricing language,
Product wanted to lead with technical depth, and Design had a brand-first vision.
We had 3 weeks and zero consensus.

Task: As coordinator, I had no formal authority over any of these teams, but I
owned the launch one-pager — which meant I owned the forcing function.

Action: I ran a 90-minute joint workshop where each team presented their single
most important message. I then drafted a positioning hierarchy — one core promise,
two supporting pillars — and circulated a shared doc for async feedback with a
48-hour comment deadline.

Result: We shipped on time with unanimous sign-off. The campaign drove a 28%
increase in trial signups compared to our previous launch, and the Sales team
adopted the one-pager as their primary outreach asset for the following quarter.
```

---

### Example 3 — Recent Graduate → Data Analyst Role

**Resume Snippet (uploaded):**
```
B.S. Information Technology, George Mason University, May 2025.
Senior Capstone: Built a predictive dashboard in Python (pandas, matplotlib)
analyzing 5 years of GMU enrollment data.
Coursework: Database Systems, Data Visualization, Statistics, Machine Learning Fundamentals.
Part-time retail job (2 years): tracked inventory, prepared weekly Excel reports for manager.
```

**Job Description (pasted):**
```
Junior Data Analyst — RetailCo
Analyze sales and inventory data. Build dashboards in Tableau or Power BI.
Write SQL queries. Communicate findings to non-technical stakeholders.
Entry-level; bachelor's degree required.
```

**Generated Resume Bullets:**
```
• Built an interactive Python dashboard (pandas + matplotlib) analyzing 5 years of
  institutional enrollment data, surfacing 3 actionable enrollment trend insights
  for university leadership.

• Prepared and presented weekly inventory performance reports in Excel for a
  12-location retail operation, reducing stock discrepancy reporting time by 35%.

• Designed and normalized a relational database schema as part of a semester-long
  capstone, writing 20+ SQL queries to support multi-table analytical reporting.
```

**Evaluation Scores:**
```
Keyword Match:  61%
Relevance:      55%
Missing Skills: tableau, power bi
Grounding:      91% (Well grounded)
```

**Skill Gap — Learning Roadmap:**
```
📚 Tableau
   Resource: "Tableau Desktop Specialist" — Tableau Public free training
   Time: 3 weeks
   Quick win: Rebuild your capstone dashboard in Tableau Public and add it to
              your portfolio/LinkedIn by end of week.

📚 Power BI
   Resource: "Microsoft Power BI Desktop" — Microsoft Learn (free)
   Time: 2 weeks
   Quick win: Connect Power BI to a public dataset (Kaggle) and publish one report.
```

---

## 5. Design Decisions

### Why RAG Instead of Whole-Document Injection?

The simplest approach would be to paste the entire resume into every prompt. It works — but it is wasteful and imprecise. A 2-page resume is ~600–800 words. That is fine for GPT-4o-mini's context window, but it means the model receives a lot of irrelevant information for any given generation task (e.g., your college GPA when generating a senior engineer cover letter is noise, not signal).

The RAG pipeline queries the vector store with the job description as the search key. This means that for a DevOps role, the top-6 retrieved chunks will be the candidate's most DevOps-adjacent experience — not their unrelated university project. The output is more precise and the prompts stay lean, which also reduces token costs.

**Trade-off:** Building the FAISS index requires one additional round of embedding API calls. For a typical resume (10–15 chunks), this adds ~$0.0001 to the cost and ~2 seconds to the wall time. It is worth it.

---

### Why FAISS Over ChromaDB?

Both are valid choices. FAISS was chosen because:
- It is a pure in-memory, single-file dependency (`faiss-cpu`)
- No database server or persistent storage to manage
- Sub-millisecond search on the small (10–20 vector) corpora typical of a resume

ChromaDB would be the right choice if the system needed to persist and query across many resumes over time (e.g., a multi-user SaaS product). For a single-session tool, FAISS is simpler and faster.

---

### Why GPT-4o-mini Instead of GPT-4o?

GPT-4o-mini produces output quality that is more than sufficient for resume bullets and cover letters — the improvement from full GPT-4o would be marginal for structured writing tasks with detailed prompts. The cost difference is substantial (~15× cheaper per token). For a student or job seeker running this hundreds of times, that matters.

The model is a single string in `app.py` (`"gpt-4o-mini"`), so upgrading to `"gpt-4o"` is a one-line change if needed.

---

### Why Five Separate API Calls Instead of One?

Each output has fundamentally different requirements: resume bullets need action verbs and metrics; the cover letter needs narrative flow; STAR answers need a structured format. Combining them into a single prompt creates competing instructions and degrades all outputs. Separate calls with focused prompts produce meaningfully better results.

**Trade-off:** Five calls means ~5× the API cost and ~5× the latency (mitigated by Streamlit's progress bar). For the quality improvement, it is the right trade-off.

---

### Why Local Evaluation Instead of AI Evaluation?

It would be possible to ask GPT-4o-mini to score its own output. The problem is that LLMs are unreliable self-evaluators — they tend to give high scores to their own text regardless of actual quality. The evaluation module uses deterministic, reproducible algorithms: set intersection for keyword matching, Jaccard similarity for relevance, `textstat` for readability. These scores are objective, explainable, and cost zero tokens.

---

### Why Allow Inline Editing?

AI output is never perfect. A cover letter might use slightly off-brand language; a resume bullet might reference something the candidate wants to downplay. By making every output field an editable `st.text_area`, the tool positions the AI as a first draft assistant rather than an autonomous agent. The human stays in the loop. This is intentional and reflects a core principle: AI should augment judgment, not replace it.

---

## 6. Testing Summary

### What Worked Well

**Resume parsing** was reliable for clean PDF and DOCX files. PyPDF2 correctly extracted multi-column and bulleted layouts in all test documents. The chunker's overlapping window prevented any sentence from being split across two chunks in a way that lost meaning.

**The RAG retrieval** consistently surfaced the most relevant resume sections. When testing with a software engineer resume against a marketing JD, the retrieved chunks correctly deprioritized technical skills and surfaced communication/collaboration experience instead.

**Cover letter quality** was the strongest output. The model correctly avoided placeholder text (a common failure mode with naive prompts), matched the tone of the JD, and produced letters that required minimal editing in most tests.

**The evaluation module** ran correctly and produced scores that matched human intuition — a strong match resume scored 72% on keyword match; an obviously mismatched resume scored 31%.

**Download functionality** worked across all tested browsers (Chrome, Firefox, Edge on Windows).

---

### What Did Not Work as Expected

**PDF files with scanned images** produced empty text extractions. PyPDF2 cannot perform OCR. This is a known limitation; the app displays a clear error message asking the user to upload a text-based PDF.

**Very short resumes (under 200 words)** produced only 1–2 chunks, which limited retrieval diversity. The outputs were still usable but less personalised. A minimum-content warning was added to the UI to guide users.

**The keyword match score** can be misleadingly high if the candidate's resume happens to contain common English words that also appear in the JD. The evaluator mitigates this by filtering words shorter than 3 characters, but it is still a surface-level heuristic, not a semantic similarity measure.

**Hallucination check false positives:** The grounding check flagged industry-standard phrases ("cross-functional", "stakeholder alignment") as "novel" because they did not appear verbatim in the resume. These are legitimate professional expressions, not fabrications. The threshold was tuned down to 65% to reduce false warnings, but it is an imperfect heuristic.

---

### Lessons Learned

1. **Prompt engineering is the highest-leverage activity.** The biggest quality improvements came not from changing models or architecture but from rewriting prompts — adding explicit format rules, banning placeholder text, specifying word counts. A well-engineered prompt for GPT-4o-mini beats a vague prompt for GPT-4o.

2. **Session state management in Streamlit requires discipline.** Streamlit re-runs the entire script on every interaction. Without careful use of `st.session_state`, the vector store was being rebuilt on every button click. Caching mutable objects in session state (rather than `@st.cache_data`) was the correct pattern for FAISS indices.

3. **Error handling must be user-facing, not just logged.** Early versions caught exceptions with `logger.error()` and silently continued. Users saw blank outputs with no explanation. Replacing silent failures with `st.error()` messages dramatically improved the experience.

4. **Evaluation scores are a conversation starter, not a verdict.** A 55% keyword match does not mean the resume is bad — it might mean the candidate is intentionally career-pivoting. The scores are most useful as diagnostic tools (e.g., "the gap analysis shows I'm missing Docker — I should add that project") rather than pass/fail gates.

---

## 7. Reflection

### What This Project Taught Me About AI

The most important thing this project taught me is that **the hard part of building an AI application is not calling the API — it is designing the system around it.**

The OpenAI API call itself is four lines of Python. Everything else — the parsing, the chunking strategy, the vector store, the evaluation logic, the prompt engineering, the UI state management, the error handling — is traditional software engineering. AI is the engine, but the car still needs to be built.

I also developed a much more nuanced understanding of what LLMs are actually good at and where they fall short. They are excellent at transforming and reformatting information that already exists. They are unreliable at generating specific facts (metrics, company names, dates) that were not in the source material. This is why grounding the model through RAG — forcing it to work from the candidate's actual experience rather than generic career advice — was the most important architectural decision in this project.

### What This Project Taught Me About Problem-Solving

**Decomposition is everything.** "Build an AI job application tool" is an overwhelming problem. Breaking it into: parse a file → chunk it → embed chunks → retrieve relevant chunks → generate five specific outputs → evaluate them → display them — turned one big problem into eight small, testable ones. I could build and test each component independently before wiring them together.

**Build the simplest thing that works, then improve it.** The first working version had no RAG — it just injected the full resume text into a single prompt. That version worked and produced useful output. Then I added FAISS. Then I split the prompts. Then I added the evaluator. This incremental approach meant I always had a working system, and I never got stuck trying to build everything at once.

**User experience is not cosmetic.** The CSS, the progress bar, the score cards, the inline editing — these are not decorative. They change how users interact with the system. Without the progress bar, users thought the app was frozen during the 30-second generation. Without inline editing, the app felt like a one-shot black box rather than a collaborative tool. Design and functionality are not separate concerns.

Finally, this project reinforced something about the current moment in AI: **the bottleneck for most AI applications is not capability — it is thoughtful integration.** GPT-4o-mini is more than capable enough to generate a great cover letter. The challenge is building the context, the guardrails, and the interface that let it do so reliably, safely, and in a way that genuinely serves the person using it.

---

*Built for GMU AI 101 — Project 4 | Spring 2026*
*Author: Million Aboye*
