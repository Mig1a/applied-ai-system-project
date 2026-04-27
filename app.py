"""
AI Job Copilot — Main Streamlit Application
============================================
A full-stack AI tool that helps users apply for jobs faster and smarter.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

from utils.parser import extract_resume_text, chunk_text
from utils.rag import build_vector_store, retrieve_relevant_chunks, build_context
from utils.evaluator import run_full_evaluation
from utils.prompts import (
    get_system_prompt,
    get_resume_bullets_prompt,
    get_cover_letter_prompt,
    get_interview_questions_prompt,
    get_star_answers_prompt,
    get_skill_gap_prompt,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

# Write logs to file AND console so failures are always recorded
Path("outputs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("AI Job Copilot started.")

st.set_page_config(
    page_title="AI Job Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] .stRadio label { cursor: pointer; }

    /* ---- Score cards ---- */
    .score-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        border: 1px solid #334155;
    }
    .score-value {
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-label {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .score-green  { color: #22c55e; }
    .score-yellow { color: #f59e0b; }
    .score-red    { color: #ef4444; }

    /* ---- Info banner ---- */
    .info-banner {
        background: #0f172a;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    /* ---- Pill badges ---- */
    .pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 3px;
    }
    .pill-green  { background: #14532d; color: #86efac; }
    .pill-red    { background: #450a0a; color: #fca5a5; }
    .pill-yellow { background: #451a03; color: #fcd34d; }

    /* ---- Output box ---- */
    .output-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.7;
        color: #e2e8f0;
    }

    /* ---- Step badge ---- */
    .step-badge {
        background: #6366f1;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 8px;
    }

    /* ---- Divider ---- */
    hr { border-color: #334155; }

    /* ---- Streamlit overrides ---- */
    .stTextArea textarea { background: #1e293b; color: #f1f5f9; border: 1px solid #475569; }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
    .stDownloadButton > button {
        background: #1e293b;
        color: #a5b4fc;
        border: 1px solid #4f46e5;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

DEFAULTS = {
    "resume_text": "",
    "resume_chunks": [],
    "vector_store": None,
    "store_chunks": [],
    "jd_text": "",
    "cover_letter_draft": "",
    "outputs": {},
    "confidence_scores": {},
    "evaluation": {},
    "feedback": {},
    "generated": False,
    "api_key_valid": False,
    "openai_client": None,
    "current_page": "Home",
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client() -> Optional[OpenAI]:
    key = st.session_state.get("api_key_override") or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    return OpenAI(api_key=key)


def _colour_class(score: float, thresholds=(60, 40)) -> str:
    high, low = thresholds
    if score >= high:
        return "score-green"
    if score >= low:
        return "score-yellow"
    return "score-red"


def _score_card(label: str, value, suffix: str = "") -> str:
    try:
        colour = _colour_class(float(value))
    except (TypeError, ValueError):
        colour = "score-yellow"
    return (
        f'<div class="score-card">'
        f'  <div class="score-value {colour}">{value}{suffix}</div>'
        f'  <div class="score-label">{label}</div>'
        f"</div>"
    )


def _call_openai(client: OpenAI, user_prompt: str, max_tokens: int = 1_500) -> str:
    """Thin wrapper around ChatCompletion with logging."""
    logger.info("OpenAI call — prompt length: %d chars, max_tokens: %d", len(user_prompt), max_tokens)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        result = response.choices[0].message.content.strip()
        logger.info("OpenAI call succeeded — response length: %d chars", len(result))
        return result
    except Exception as exc:
        logger.error("OpenAI call failed: %s", exc)
        raise


def _score_confidence(client: OpenAI, output_name: str, content: str, context: str) -> int:
    """
    Ask GPT-4o-mini to rate its own confidence in the generated output (1-10).
    Returns an integer score. Falls back to 5 on any error.
    """
    prompt = (
        f"You just generated the following {output_name} for a job application:\n\n"
        f"{content[:800]}\n\n"
        f"The candidate context available was:\n{context[:400]}\n\n"
        f"On a scale of 1-10, how confident are you that this output is accurate, "
        f"relevant, and well-grounded in the candidate's actual experience? "
        f"Reply with ONLY a single integer between 1 and 10."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        score = int("".join(filter(str.isdigit, raw)))
        logger.info("Confidence score for %s: %d/10", output_name, score)
        return max(1, min(10, score))
    except Exception as exc:
        logger.warning("Confidence scoring failed for %s: %s", output_name, exc)
        return 5


def _save_output(filename: str, content: str) -> str:
    """Write content to the outputs/ directory, return the path."""
    path = Path("outputs") / filename
    path.write_text(content, encoding="utf-8")
    logger.info("Saved output to %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# Core generation pipeline
# ---------------------------------------------------------------------------

def run_generation_pipeline(
    client: OpenAI,
    resume_text: str,
    jd_text: str,
    cover_letter_draft: str,
    progress_bar,
    status_text,
) -> Dict:
    """
    Full RAG + generation pipeline.
    Returns a dict with all generated outputs.
    """
    outputs: Dict = {}
    confidence: Dict = {}

    logger.info("Pipeline started — resume length: %d chars, JD length: %d chars",
                len(resume_text), len(jd_text))

    # 1. Build vector store
    status_text.markdown("**Step 1/6 — Building vector store from resume…**")
    chunks = chunk_text(resume_text)
    logger.info("Chunked resume into %d chunks", len(chunks))
    index, store_chunks = build_vector_store(chunks, client)
    st.session_state["vector_store"] = index
    st.session_state["store_chunks"] = store_chunks
    progress_bar.progress(15)

    # 2. Retrieve resume context relevant to the JD
    status_text.markdown("**Step 2/6 — Retrieving relevant resume sections…**")
    relevant = retrieve_relevant_chunks(jd_text, index, store_chunks, client, k=6)
    context = build_context(relevant, max_words=2_000)
    logger.info("Retrieved %d relevant chunks", len(relevant))
    progress_bar.progress(28)

    # 3. Resume bullet points
    status_text.markdown("**Step 3/6 — Generating ATS-optimised resume bullets…**")
    bullets_prompt = get_resume_bullets_prompt(context, jd_text)
    outputs["resume_bullets"] = _call_openai(client, bullets_prompt, max_tokens=800)
    confidence["resume_bullets"] = _score_confidence(client, "resume bullets", outputs["resume_bullets"], context)
    progress_bar.progress(43)

    # 4. Cover letter
    status_text.markdown("**Step 4/6 — Writing tailored cover letter…**")
    cl_prompt = get_cover_letter_prompt(context, jd_text, cover_letter_draft or None)
    outputs["cover_letter"] = _call_openai(client, cl_prompt, max_tokens=700)
    confidence["cover_letter"] = _score_confidence(client, "cover letter", outputs["cover_letter"], context)
    progress_bar.progress(58)

    # 5. Interview questions + STAR answers
    status_text.markdown("**Step 5/6 — Generating interview questions & STAR answers…**")
    iq_prompt = get_interview_questions_prompt(jd_text, context)
    outputs["interview_questions"] = _call_openai(client, iq_prompt, max_tokens=600)
    star_prompt = get_star_answers_prompt(context, outputs["interview_questions"])
    outputs["star_answers"] = _call_openai(client, star_prompt, max_tokens=1_500)
    confidence["interview_questions"] = _score_confidence(client, "interview questions", outputs["interview_questions"], context)
    confidence["star_answers"] = _score_confidence(client, "STAR answers", outputs["star_answers"], context)
    progress_bar.progress(75)

    # 6. Skill gap analysis
    status_text.markdown("**Step 6/6 — Analysing skill gaps…**")
    gap_prompt = get_skill_gap_prompt(resume_text, jd_text)
    outputs["skill_gap"] = _call_openai(client, gap_prompt, max_tokens=900)
    confidence["skill_gap"] = _score_confidence(client, "skill gap analysis", outputs["skill_gap"], context)
    progress_bar.progress(92)

    logger.info("Pipeline complete. Confidence scores: %s", confidence)
    st.session_state["confidence_scores"] = confidence
    return outputs


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 20px;">
            <span style="font-size:2.4rem;">🚀</span>
            <h2 style="margin:4px 0 2px; font-size:1.4rem; font-weight:700;">AI Job Copilot</h2>
            <p style="font-size:0.78rem; color:#94a3b8; margin:0;">Powered by GPT-4o-mini + RAG</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "⚙️ Settings", "📖 How It Works"],
        label_visibility="collapsed",
    )
    st.session_state["current_page"] = page

    st.divider()

    # Status indicators
    st.markdown("**Status**")
    api_ok = bool(st.session_state.get("api_key_override") or os.getenv("OPENAI_API_KEY"))
    resume_ok = bool(st.session_state["resume_text"])
    jd_ok = bool(st.session_state["jd_text"])
    generated_ok = st.session_state["generated"]

    def _status(label, ok):
        icon = "✅" if ok else "⭕"
        st.markdown(f"{icon} {label}")

    _status("API key configured", api_ok)
    _status("Resume uploaded", resume_ok)
    _status("Job description added", jd_ok)
    _status("Outputs generated", generated_ok)

    st.divider()
    st.markdown(
        '<p style="font-size:0.72rem; color:#64748b; text-align:center;">'
        "AI Job Copilot v1.0<br>GMU AI 101 — Project 4</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Settings
# ---------------------------------------------------------------------------

if page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("Configure your OpenAI API key. The key is stored only in your session.")

    with st.form("settings_form"):
        key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            value=st.session_state.get("api_key_override", ""),
            help="Get your key at platform.openai.com",
        )
        saved = st.form_submit_button("Save Key", use_container_width=True)

    if saved:
        if key_input.startswith("sk-"):
            st.session_state["api_key_override"] = key_input
            # Quick validation
            try:
                test_client = OpenAI(api_key=key_input)
                test_client.models.list()
                st.success("✅ API key is valid and saved for this session.")
            except AuthenticationError:
                st.error("❌ Invalid API key. Please check and try again.")
            except Exception:
                st.warning("⚠️ Key saved, but we couldn't validate it right now.")
        else:
            st.error("API keys should start with `sk-`.")

    st.divider()
    st.markdown("**Environment variable fallback**")
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        st.success(f"✅ `OPENAI_API_KEY` found in `.env` (ends in …{env_key[-4:]})")
    else:
        st.warning("No `OPENAI_API_KEY` found in environment. Use the field above or add one to `.env`.")


# ---------------------------------------------------------------------------
# Page: How It Works
# ---------------------------------------------------------------------------

elif page == "📖 How It Works":
    st.title("📖 How AI Job Copilot Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### Architecture

```
Resume (PDF/DOCX)
        │
        ▼
  Document Parser         ← PyPDF2 / python-docx
        │
        ▼
   Text Chunker            ← 400-word overlapping chunks
        │
        ▼
 OpenAI Embeddings         ← text-embedding-3-small
        │
        ▼
  FAISS Vector Store       ← cosine similarity index
        │
    ┌───┴───┐
    │       │
    ▼       ▼
  RAG    Job Description
 Query      │
    │       │
    └───┬───┘
        │
        ▼
  GPT-4o-mini Generator    ← 5 tailored outputs
        │
        ▼
    Evaluator              ← scoring + gap analysis
        │
        ▼
  Output Dashboard         ← edit / copy / download
```
""")

    with col2:
        st.markdown("""
### What Gets Generated

| Output | Description |
|---|---|
| **Resume Bullets** | 7 ATS-optimised bullet points with metrics |
| **Cover Letter** | Personalised, keyword-rich, ~300 words |
| **Interview Questions** | 10 role-specific questions across 4 categories |
| **STAR Answers** | Full answers using Situation-Task-Action-Result |
| **Skill Gap Analysis** | Matched skills, gaps, and a learning roadmap |

### Evaluation Scores

| Score | What it measures |
|---|---|
| **Keyword Match** | % of JD keywords present in your resume |
| **Relevance Score** | Jaccard similarity between resume and JD |
| **Readability** | Flesch Reading Ease of generated content |
| **Grounding** | How grounded the cover letter is in your actual experience |

### Privacy
Your resume and API key never leave your machine.
Everything runs locally and calls the OpenAI API directly.
""")


# ---------------------------------------------------------------------------
# Page: Home (main workflow)
# ---------------------------------------------------------------------------

else:  # Home
    st.markdown(
        """
        <div style="padding:10px 0 4px">
            <h1 style="font-size:2rem; font-weight:800; margin:0;">
                🚀 AI Job Copilot
            </h1>
            <p style="color:#94a3b8; margin:4px 0 0;">
                Upload your resume, paste a job description, and get AI-powered application materials in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Input Section -------------------------------------------------- #

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<span class="step-badge">1</span> **Upload Your Resume**', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Supported formats: PDF, DOCX",
            type=["pdf", "docx"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            with st.spinner("Extracting resume text…"):
                try:
                    resume_text = extract_resume_text(uploaded_file)
                    st.session_state["resume_text"] = resume_text
                    st.session_state["resume_chunks"] = chunk_text(resume_text)
                    st.success(
                        f"✅ {uploaded_file.name} — {len(resume_text):,} characters extracted."
                    )
                    with st.expander("Preview extracted text"):
                        st.text(resume_text[:1_500] + ("…" if len(resume_text) > 1_500 else ""))
                except ValueError as exc:
                    st.error(f"❌ {exc}")

        st.markdown('<span class="step-badge">3</span> **Cover Letter Draft** *(optional)*', unsafe_allow_html=True)
        cover_letter_draft = st.text_area(
            "Paste an existing draft to improve, or leave blank",
            height=130,
            placeholder="Optional: paste your existing draft here and the AI will refine it…",
            label_visibility="collapsed",
        )
        st.session_state["cover_letter_draft"] = cover_letter_draft

    with col_right:
        st.markdown('<span class="step-badge">2</span> **Paste Job Description**', unsafe_allow_html=True)
        jd_text = st.text_area(
            "Paste the full job posting here",
            height=320,
            placeholder="Paste the complete job description here…\n\nTip: include the full posting with responsibilities and requirements for best results.",
            label_visibility="collapsed",
        )
        st.session_state["jd_text"] = jd_text

    st.divider()

    # ---- Generate Button ------------------------------------------------ #

    col_btn, col_info = st.columns([1, 2])

    with col_btn:
        generate_clicked = st.button(
            "⚡ Analyse & Generate",
            use_container_width=True,
            disabled=not (
                st.session_state["resume_text"] and st.session_state["jd_text"]
            ),
        )

    with col_info:
        if not st.session_state["resume_text"]:
            st.markdown(
                '<div class="info-banner">⬆️ Upload your resume to get started.</div>',
                unsafe_allow_html=True,
            )
        elif not st.session_state["jd_text"]:
            st.markdown(
                '<div class="info-banner">📋 Paste a job description to enable generation.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="info-banner">✅ Ready! Click <strong>Analyse & Generate</strong> to run the full pipeline.</div>',
                unsafe_allow_html=True,
            )

    # ---- Generation Logic ---------------------------------------------- #

    if generate_clicked:
        client = _get_client()
        if not client:
            st.error(
                "❌ No OpenAI API key found. Go to ⚙️ Settings to add one, "
                "or add `OPENAI_API_KEY=sk-...` to your `.env` file."
            )
        else:
            progress_bar = st.progress(0, text="Starting pipeline…")
            status_text = st.empty()

            try:
                with st.spinner("Running AI pipeline — this takes ~30 seconds…"):
                    outputs = run_generation_pipeline(
                        client=client,
                        resume_text=st.session_state["resume_text"],
                        jd_text=st.session_state["jd_text"],
                        cover_letter_draft=st.session_state["cover_letter_draft"],
                        progress_bar=progress_bar,
                        status_text=status_text,
                    )

                st.session_state["outputs"] = outputs
                status_text.markdown("**Evaluating outputs…**")
                evaluation = run_full_evaluation(
                    st.session_state["resume_text"],
                    st.session_state["jd_text"],
                    outputs,
                )
                st.session_state["evaluation"] = evaluation
                st.session_state["generated"] = True
                progress_bar.progress(100, text="Done!")
                status_text.empty()
                st.success("✅ All outputs generated successfully! Scroll down to review.")

            except AuthenticationError:
                st.error("❌ Invalid API key. Please update it in ⚙️ Settings.")
                progress_bar.empty()
                status_text.empty()
            except RateLimitError:
                st.error("❌ OpenAI rate limit hit. Wait a moment and try again.")
                progress_bar.empty()
                status_text.empty()
            except Exception as exc:
                st.error(f"❌ Generation failed: {exc}")
                logger.exception("Pipeline error")
                progress_bar.empty()
                status_text.empty()

    # ---- Results Section ----------------------------------------------- #

    if st.session_state["generated"] and st.session_state["outputs"]:
        st.divider()
        st.markdown("## 📊 Results Dashboard")

        # Score cards row
        eval_data = st.session_state["evaluation"]
        confidence_scores = st.session_state.get("confidence_scores", {})
        keyword_score = eval_data.get("keyword_match", {}).get("score", 0)
        relevance = eval_data.get("relevance_score", 0)
        missing_count = eval_data.get("skill_analysis", {}).get("missing_count", 0)
        grounding = eval_data.get("hallucination_check", {}).get("grounding_score", 100)
        avg_confidence = (
            round(sum(confidence_scores.values()) / len(confidence_scores), 1)
            if confidence_scores else "—"
        )

        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.markdown(_score_card("Keyword Match", f"{keyword_score:.0f}", "%"), unsafe_allow_html=True)
        with s2:
            st.markdown(_score_card("Relevance Score", f"{relevance:.0f}", "%"), unsafe_allow_html=True)
        with s3:
            st.markdown(_score_card("Missing Skills", missing_count), unsafe_allow_html=True)
        with s4:
            st.markdown(_score_card("Grounding Score", f"{grounding:.0f}", "%"), unsafe_allow_html=True)
        with s5:
            st.markdown(_score_card("AI Confidence", f"{avg_confidence}", "/10"), unsafe_allow_html=True)

        # Confidence breakdown
        if confidence_scores:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🤖 AI Self-Confidence Breakdown", expanded=False):
                st.caption("The AI rated how confident it was in each output based on the context provided (1 = low, 10 = high).")
                cc = st.columns(len(confidence_scores))
                for i, (name, score) in enumerate(confidence_scores.items()):
                    label = name.replace("_", " ").title()
                    colour = "score-green" if score >= 7 else ("score-yellow" if score >= 5 else "score-red")
                    cc[i].markdown(
                        f'<div class="score-card">'
                        f'<div class="score-value {colour}">{score}/10</div>'
                        f'<div class="score-label">{label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # Main output tabs
        tabs = st.tabs([
            "📝 Resume Bullets",
            "✉️ Cover Letter",
            "🎤 Interview Prep",
            "📚 STAR Answers",
            "🔍 Skill Gap",
            "📈 Evaluation",
        ])

        outputs = st.session_state["outputs"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---- Tab 1: Resume Bullets ---- #
        with tabs[0]:
            st.markdown("### ATS-Optimised Resume Bullets")
            st.caption("Edit the text below, then copy or download.")
            edited_bullets = st.text_area(
                "Resume Bullets",
                value=outputs.get("resume_bullets", ""),
                height=340,
                key="edit_bullets",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download as .txt",
                    data=edited_bullets,
                    file_name=f"resume_bullets_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col2:
                if st.button("💾 Save to outputs/", key="save_bullets", use_container_width=True):
                    _save_output(f"resume_bullets_{timestamp}.txt", edited_bullets)
                    st.success("Saved to outputs/ folder.")

        # ---- Tab 2: Cover Letter ---- #
        with tabs[1]:
            st.markdown("### Tailored Cover Letter")
            st.caption("Professional, keyword-rich, ~300 words. Edit freely.")
            edited_cl = st.text_area(
                "Cover Letter",
                value=outputs.get("cover_letter", ""),
                height=420,
                key="edit_cl",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download as .txt",
                    data=edited_cl,
                    file_name=f"cover_letter_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col2:
                if st.button("💾 Save to outputs/", key="save_cl", use_container_width=True):
                    _save_output(f"cover_letter_{timestamp}.txt", edited_cl)
                    st.success("Saved to outputs/ folder.")

            # Readability inline
            read_data = eval_data.get("readability", {}).get("cover_letter", {})
            if read_data:
                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                r1.metric("Flesch Score", read_data.get("flesch_score", "—"))
                r2.metric("Grade Level", read_data.get("grade_level", "—"))
                r3.metric("Readability", read_data.get("readability", "—"))

        # ---- Tab 3: Interview Questions ---- #
        with tabs[2]:
            st.markdown("### Interview Questions")
            st.caption("10 role-specific questions across Behavioral, Technical, Situational, and Culture Fit.")
            edited_iq = st.text_area(
                "Interview Questions",
                value=outputs.get("interview_questions", ""),
                height=380,
                key="edit_iq",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download as .txt",
                    data=edited_iq,
                    file_name=f"interview_questions_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col2:
                if st.button("💾 Save to outputs/", key="save_iq", use_container_width=True):
                    _save_output(f"interview_questions_{timestamp}.txt", edited_iq)
                    st.success("Saved.")

        # ---- Tab 4: STAR Answers ---- #
        with tabs[3]:
            st.markdown("### STAR Method Answers")
            st.caption("Structured answers: Situation → Task → Action → Result. Edit to personalise further.")
            edited_star = st.text_area(
                "STAR Answers",
                value=outputs.get("star_answers", ""),
                height=560,
                key="edit_star",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download as .txt",
                    data=edited_star,
                    file_name=f"star_answers_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col2:
                if st.button("💾 Save to outputs/", key="save_star", use_container_width=True):
                    _save_output(f"star_answers_{timestamp}.txt", edited_star)
                    st.success("Saved.")

        # ---- Tab 5: Skill Gap ---- #
        with tabs[4]:
            st.markdown("### Skill Gap Analysis")

            skill_data = eval_data.get("skill_analysis", {})

            gcol, mcol = st.columns(2)
            with gcol:
                st.markdown("**✅ Skills You Have (matching JD)**")
                for skill in skill_data.get("present_skills", []):
                    st.markdown(
                        f'<span class="pill pill-green">{skill}</span>',
                        unsafe_allow_html=True,
                    )
            with mcol:
                st.markdown("**❌ Skills to Develop**")
                for skill in skill_data.get("missing_skills", []):
                    st.markdown(
                        f'<span class="pill pill-red">{skill}</span>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown("**📚 Full AI Skill Gap Report**")
            edited_gap = st.text_area(
                "Skill Gap",
                value=outputs.get("skill_gap", ""),
                height=420,
                key="edit_gap",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download as .txt",
                    data=edited_gap,
                    file_name=f"skill_gap_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col2:
                if st.button("💾 Save to outputs/", key="save_gap", use_container_width=True):
                    _save_output(f"skill_gap_{timestamp}.txt", edited_gap)
                    st.success("Saved.")

        # ---- Tab 6: Evaluation ---- #
        with tabs[5]:
            st.markdown("### Evaluation Report")

            # Keyword match details
            kw = eval_data.get("keyword_match", {})
            st.markdown("#### Keyword Analysis")
            kcol1, kcol2, kcol3 = st.columns(3)
            kcol1.metric("Keyword Match Score", f"{kw.get('score', 0):.1f}%")
            kcol2.metric("Matched Keywords", kw.get("matched_count", 0))
            kcol3.metric("Total JD Keywords", kw.get("total_jd_keywords", 0))

            if kw.get("matched_keywords"):
                st.markdown("**Matched keywords:**")
                pills = "".join(
                    f'<span class="pill pill-green">{k}</span>'
                    for k in kw["matched_keywords"][:25]
                )
                st.markdown(pills, unsafe_allow_html=True)

            st.markdown("---")

            # Skill analysis
            sa = eval_data.get("skill_analysis", {})
            st.markdown("#### Skill Coverage")
            sc1, sc2 = st.columns(2)
            sc1.metric("Present Skills", sa.get("present_count", 0))
            sc2.metric("Missing Skills", sa.get("missing_count", 0))

            st.markdown("---")

            # Readability table
            read_all = eval_data.get("readability", {})
            if read_all:
                st.markdown("#### Readability Scores")
                import pandas as pd
                rows = []
                for output_name, r in read_all.items():
                    rows.append({
                        "Output": output_name.replace("_", " ").title(),
                        "Flesch Score": r.get("flesch_score", "—"),
                        "Grade Level": r.get("grade_level", "—"),
                        "Readability": r.get("readability", "—"),
                        "Word Count": r.get("word_count", "—"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("---")

            # Hallucination check
            hc = eval_data.get("hallucination_check", {})
            if hc:
                st.markdown("#### Cover Letter Grounding Check")
                hc1, hc2, hc3 = st.columns(3)
                hc1.metric("Grounding Score", f"{hc.get('grounding_score', 0):.1f}%")
                hc2.metric("Novel Words", hc.get("novel_words_count", 0))
                hc3.metric("Status", hc.get("status", "—"))
                if hc.get("warning"):
                    st.warning(
                        "⚠️ The cover letter contains significant content not directly "
                        "from your resume or the job description. Please review carefully."
                    )

            st.markdown("---")

            # Download full evaluation report
            import json
            eval_json = json.dumps(eval_data, indent=2)
            st.download_button(
                "⬇️ Download Full Evaluation Report (JSON)",
                data=eval_json,
                file_name=f"evaluation_report_{timestamp}.json",
                mime="application/json",
                use_container_width=False,
            )

        # ---- Human Feedback Panel ---- #
        st.divider()
        st.markdown("### 🧑‍⚖️ Human Evaluation")
        st.caption("Rate each output. Your feedback is saved to `outputs/feedback.json` for future review.")

        OUTPUTS_TO_RATE = {
            "resume_bullets": "Resume Bullets",
            "cover_letter": "Cover Letter",
            "interview_questions": "Interview Questions",
            "star_answers": "STAR Answers",
            "skill_gap": "Skill Gap Analysis",
        }

        feedback: Dict = st.session_state.get("feedback", {})
        fb_cols = st.columns(len(OUTPUTS_TO_RATE))

        for i, (key, label) in enumerate(OUTPUTS_TO_RATE.items()):
            with fb_cols[i]:
                st.markdown(f"**{label}**")
                rating = st.radio(
                    f"Rate {label}",
                    options=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                    index=feedback.get(key, {}).get("stars", 2),
                    key=f"rating_{key}",
                    label_visibility="collapsed",
                )
                feedback.setdefault(key, {})["stars"] = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"].index(rating)
                feedback[key]["label"] = rating

        st.markdown("<br>", unsafe_allow_html=True)
        overall_notes = st.text_area(
            "Overall notes (optional)",
            placeholder="Any comments on the quality, accuracy, or usefulness of the outputs…",
            height=80,
            key="feedback_notes",
        )

        if st.button("💾 Save Feedback", use_container_width=False):
            import json
            feedback["overall_notes"] = overall_notes
            feedback["timestamp"] = datetime.now().isoformat()
            feedback["confidence_scores"] = confidence_scores
            st.session_state["feedback"] = feedback
            _save_output("feedback.json", json.dumps(feedback, indent=2))
            logger.info("Human feedback saved: %s", feedback)
            st.success("✅ Feedback saved to outputs/feedback.json")

        # ---- Download All Bundle ---- #
        st.divider()
        st.markdown("### 📦 Download All Outputs")

        all_text = f"""AI JOB COPILOT — FULL OUTPUT BUNDLE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

RESUME BULLETS
{'='*60}
{outputs.get('resume_bullets', '')}

{'='*60}
COVER LETTER
{'='*60}
{outputs.get('cover_letter', '')}

{'='*60}
INTERVIEW QUESTIONS
{'='*60}
{outputs.get('interview_questions', '')}

{'='*60}
STAR METHOD ANSWERS
{'='*60}
{outputs.get('star_answers', '')}

{'='*60}
SKILL GAP ANALYSIS
{'='*60}
{outputs.get('skill_gap', '')}
"""
        st.download_button(
            "⬇️ Download Complete Bundle (.txt)",
            data=all_text,
            file_name=f"job_copilot_bundle_{timestamp}.txt",
            mime="text/plain",
        )
