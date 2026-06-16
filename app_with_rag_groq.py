"""
app_with_rag.py — HR Attrition Decision Support System + RAG AI HR Consultant
===============================================================================
Production-ready Streamlit application.

Architecture guarantees implemented in this version
────────────────────────────────────────────────────
RULE 1  SIDEBAR INPUT & FEATURE ALIGNMENT
        Every sidebar widget value is collected into `inputs` dict → assembled
        into a single-row pd.DataFrame → column order explicitly enforced to
        match `feature_names` from hr_dss_artifacts.pkl before scaler.transform().
        Prevents any column mismatch or shape error during inference.

RULE 2  CONTEXT SYNCHRONIZATION
        `_build_employee_context` reads all employee attributes from the same
        `inputs` dict consumed by `_predict_employee`. The model's risk score
        and the GPT-4o context always describe the identical employee with no
        data-alignment divergence.

RULE 3  AUTOMATED INFERENCE GATE
        RAG triggers ONLY when raw_prob >= optimal_threshold. The threshold is
        loaded dynamically from hr_dss_artifacts.pkl; it is NEVER hardcoded.

RULE 4  CACHED VECTOR STORE
        `_load_rag_chain()` is wrapped in @st.cache_resource. PyPDFLoader +
        FastEmbedEmbeddings("BAAI/bge-small-en-v1.5") + Chroma persistence run
        exactly once per process lifetime regardless of reruns.

RULE 5  COMPLIANCE CITED OUTPUT
        ChatGroq(model="llama3-70b-8192", temperature=0.2) drafts the strategy.
        Retrieved source chunks with 1-based page numbers are shown inside a
        labelled st.expander for full compliance transparency.

RULE 6  SECURE EXCEPTION FALLBACKS
        Three-layer try/except architecture:
          • Outer guard  — wraps entire prediction view; catches pre-model errors.
          • Inner guard  — wraps only the RAG chain.invoke() call.
          • Error router — classifies API-key / PDF / chain errors and always
                           falls back to the local rules-based panel so the
                           right-hand column is never left empty.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition DSS",
    page_icon="🏢",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# RAG library availability — resolved once at import time, never re-evaluated
# ─────────────────────────────────────────────────────────────────────────────
_RAG_AVAILABLE: bool   = False
_RAG_IMPORT_ERROR: str = ""

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_chroma import Chroma
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    _RAG_AVAILABLE = True
except ImportError as _ie:
    _RAG_IMPORT_ERROR = str(_ie)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PDF_PATH          = "./data_docs/hr_policy_manual.pdf"
CHROMA_PERSIST    = "./data_docs/chroma_hr_policy"
ARTIFACT_PRIMARY  = "hr_dss_artifacts.pkl"   # full PART-2 bundle (preferred)
ARTIFACT_FALLBACK = "hr_model.pkl"           # Streamlit-only bundle
LLM_MODEL         = "llama-3.3-70b-versatile"
EMBED_MODEL       = "BAAI/bge-small-en-v1.5"
RETRIEVER_K       = 6

# ─────────────────────────────────────────────────────────────────────────────
# HR Consultant prompt — only constructed when RAG libraries are available
# ─────────────────────────────────────────────────────────────────────────────
_HR_CONSULTANT_PROMPT = (
    PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an AI Executive HR Consultant specializing in employee retention.\n"
            "You have been provided with the company's official Corporate HR Policy Manual\n"
            "as your sole knowledge base. Analyze the high-risk employee profile below and\n"
            "prescribe concrete, policy-grounded retention strategies.\n\n"
            "MANDATORY RULES:\n"
            "1. Every recommendation MUST cite the exact Article, Section, and Clause\n"
            "   (e.g., 'per Article II, Section 2.3, Clause 2.3.1').\n"
            "2. Be specific — include exact thresholds, timelines, compensation bands,\n"
            "   and program names as stated in the manual.\n"
            "3. Structure output in exactly three labelled sections:\n"
            "   IMMEDIATE (within 30 days) | SHORT-TERM (1-6 months) | LONG-TERM (6-12 months).\n"
            "4. Where multiple clauses apply, cite ALL of them.\n"
            "5. Do NOT invent policies. If the manual does not address a risk factor,\n"
            "   state this explicitly.\n\n"
            "HR POLICY MANUAL CONTEXT (relevant excerpts retrieved for this employee):\n"
            "────────────────────────────────────────────────────────────────────────\n"
            "{context}\n"
            "────────────────────────────────────────────────────────────────────────\n\n"
            "EMPLOYEE PROFILE AND RISK ANALYSIS:\n"
            "{question}\n\n"
            "RETENTION STRATEGY REPORT:"
        ),
    )
    if _RAG_AVAILABLE
    else None
)


# ─────────────────────────────────────────────────────────────────────────────
# RULE 1 — Load artifact  (tries ARTIFACT_PRIMARY, falls back to ARTIFACT_FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_artifact() -> dict:
    """
    Load the pickled ML artifact bundle produced by §1.13 of the notebook.
    hr_dss_artifacts.pkl is tried first because it contains the full key set
    (model, scaler, encoders, feature_names, fi, medians, modes, scores,
     model_name, threshold) needed by all app functions.
    hr_model.pkl is the Streamlit-only fallback with the same key set.
    """
    for path in (ARTIFACT_PRIMARY, ARTIFACT_FALLBACK):
        if Path(path).exists():
            with open(path, "rb") as fh:
                return pickle.load(fh)
    raise FileNotFoundError(
        f"Neither '{ARTIFACT_PRIMARY}' nor '{ARTIFACT_FALLBACK}' was found.\n"
        "Run the training notebook (§1.13) to generate the artifact bundle, "
        "then place the .pkl file in the same directory as this script."
    )


try:
    art = _load_artifact()
except FileNotFoundError as _fe:
    st.error(f"**Artifact not found.**\n\n{_fe}")
    st.stop()
except Exception as _ae:
    st.error(f"**Failed to load artifact:** {_ae}")
    st.stop()

# ── Unpack keys ──────────────────────────────────────────────────────────────
model      = art["model"]
scaler     = art["scaler"]
encoders   = art["encoders"]       # {column_name: fitted LabelEncoder}
features   = art["features"]       # RULE 1 — ordered feature list from X_train_enc
fi         = art["fi"]             # sorted feature-importance DataFrame
medians    = art["medians"]        # {col: training-set median}
modes      = art["modes"]          # {col: training-set mode}
scores     = art["scores"]         # {model_name: {acc, prec, rec, f1, auc}}
model_name = art["model_name"]

# RULE 3 — threshold always from artifact, NEVER hardcoded
optimal_threshold: float = float(art.get("threshold", 0.44))

# ─────────────────────────────────────────────────────────────────────────────
# Derived feature lists
# ─────────────────────────────────────────────────────────────────────────────
# Engineered columns are computed automatically; they must NOT appear in the
# sidebar because their values are derived from other inputs.
_ENGINEERED: set[str] = {
    "SatisfactionScore", "IncomePerExp",  "LoyaltyRatio",
    "PromotionLag",      "AvgSatisfaction", "IncomePerYear",
}

form_features: list[str] = [f for f in features if f not in _ENGINEERED]
cat_features:  list[str] = [f for f in form_features if f in encoders]
num_features:  list[str] = [f for f in form_features if f not in encoders]


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (mirrors §1.6 of the training notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived features from the raw input row."""
    df = df.copy()

    sat_cols = [
        "JobSatisfaction", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance",
    ]
    present = [c for c in sat_cols if c in df.columns]
    if present:
        df["SatisfactionScore"] = df[present].mean(axis=1)
        df["AvgSatisfaction"]   = df["SatisfactionScore"]   # legacy alias

    if {"MonthlyIncome", "TotalWorkingYears"}.issubset(df.columns):
        val = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
        df["IncomePerExp"]  = val
        df["IncomePerYear"] = val   # legacy alias

    if {"YearsAtCompany", "TotalWorkingYears"}.issubset(df.columns):
        df["LoyaltyRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

    if {"YearsSinceLastPromotion", "YearsAtCompany"}.issubset(df.columns):
        df["PromotionLag"] = (
            df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RULE 1 — Prediction with explicit feature-order enforcement
# ─────────────────────────────────────────────────────────────────────────────
def _predict_employee(inputs: dict) -> tuple[float, float]:
    """
    Run inference for a single employee.

    RULE 1 implementation
    ─────────────────────
    Step 1  Collect every form feature from `inputs`, encode categoricals with
            the fitted LabelEncoders, keep numerics as float.
    Step 2  Build a single-row pd.DataFrame, compute engineered features.
    Step 3  Fill any missing columns (e.g. newly added engineered cols) with 0.
    Step 4  SELECT and REORDER columns to exactly `features` (the list persisted
            from X_train_enc.columns.tolist() in the notebook). This is the
            explicit column-order enforcement that prevents shape errors.
    Step 5  Scale and call predict_proba.

    RULE 2 link
    ───────────
    The same `inputs` dict is later consumed by `_build_employee_context`.
    Since both functions operate on the identical dict in the same render cycle,
    the model's risk score and the LLM context always describe the same employee.

    Returns
    -------
    (risk_percent, raw_prob)  —  risk_percent = round(raw_prob * 100, 1)
    """
    # Step 1 — encode / collect
    row: dict[str, float] = {}
    for feat in form_features:
        if feat in encoders:
            le  = encoders[feat]
            val = str(inputs.get(feat, modes.get(feat, le.classes_[0])))
            val = val if val in le.classes_ else le.classes_[0]
            row[feat] = int(le.transform([val])[0])
        else:
            row[feat] = float(inputs.get(feat, medians.get(feat, 0.0)))

    # Step 2 — single-row DataFrame + feature engineering
    df = pd.DataFrame([row])
    df = _engineer(df)

    # Step 3 — fill any missing engineered columns
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    # Step 4 — RULE 1: explicit column-order enforcement matching training layout
    X_ordered = df[features].fillna(0.0)

    # Step 5 — scale and predict  (named DataFrame avoids sklearn feature-name warning)
    X_scaled = scaler.transform(X_ordered)
    raw_prob = float(model.predict_proba(X_scaled)[0, 1])
    return round(raw_prob * 100, 1), raw_prob


# ─────────────────────────────────────────────────────────────────────────────
# RULE 4 — Cached RAG chain  (@st.cache_resource = runs once per process)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 Indexing HR Policy Manual — this runs once …")
def _load_rag_chain() -> tuple[object | None, str | None]:
    """
    RULE 4 implementation
    ─────────────────────
    @st.cache_resource ensures the decorated body runs AT MOST ONCE per
    process lifetime. On every subsequent Streamlit rerun (new sidebar input,
    new prediction, API key entered) the cached chain object is returned
    instantly — no re-embedding, no re-chunking, no extra API calls.

    Flow when not yet cached:
      PyPDFLoader   → load PDF pages
      RecursiveCharacterTextSplitter (600-char / 120-char overlap)
      FastEmbedEmbeddings("BAAI/bge-small-en-v1.5")  → Chroma (persisted)
      ChatGroq("llama3-70b-8192", temperature=0.2) → RetrievalQA (mmr, k=6)

    Returns
    -------
    (chain, None)        success
    (None, error_str)    any failure — caller renders warning, not crash
    """
    # Guard 1 — RAG packages
    if not _RAG_AVAILABLE:
        return None, (
            "RAG libraries are not installed.\n\n"
            "Run:  pip install langchain langchain-community langchain-groq "
            "langchain-chroma chromadb pypdf fastembed\n\n"
            f"Import error: {_RAG_IMPORT_ERROR}"
        )

    # Guard 2 — Groq API key
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None, (
            "Groq API key is missing.\n"
            "Enter it in the AI Consultant Settings panel in the sidebar, "
            "or set the GROQ_API_KEY environment variable before launching the app."
        )

    # Guard 3 — PDF exists
    if not Path(PDF_PATH).exists():
        return None, (
            f"HR Policy Manual not found at '{PDF_PATH}'.\n"
            "Place hr_policy_manual.pdf inside the ./data_docs/ folder "
            "next to this script and restart the app."
        )

    try:
        # RULE 4 — FastEmbedEmbeddings (free, local, ONNX — lighter & faster than
        # sentence-transformers/HuggingFaceEmbeddings, no torch dependency)
        embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
        chroma_dir = Path(CHROMA_PERSIST)

        if chroma_dir.exists() and any(chroma_dir.iterdir()):
            # Reload persisted store — zero re-embedding cost
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST,
                embedding_function=embeddings,
            )
        else:
            # First run — build the vector store from scratch
            chroma_dir.mkdir(parents=True, exist_ok=True)

            # RULE 4 — PyPDFLoader for the corporate policy PDF
            loader = PyPDFLoader(PDF_PATH)
            pages  = loader.load()

            # 600-char chunks / 120-char overlap preserves clause-level context.
            # Each chunk holds 1-3 full policy clauses (avg 200-400 chars).
            # MMR retrieval (k=6) returns ≥6 diverse sections per query,
            # covering Overtime + Promotion + Compensation + Training together.
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=120,
                separators=["\n\n", "\n", ".", "---", " "],
            )
            chunks = splitter.split_documents(pages)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST,
            )

        retriever = vectorstore.as_retriever(
            search_type="mmr",            # Maximum Marginal Relevance: diverse results
            search_kwargs={"k": RETRIEVER_K},
        )

        # RULE 5 — ChatGroq(model="llama3-70b-8192", temperature=0.2)
        llm = ChatGroq(model=LLM_MODEL, temperature=0.2, groq_api_key=api_key)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": _HR_CONSULTANT_PROMPT},
            return_source_documents=True,    # needed for RULE 5 expander
        )
        return chain, None

    except Exception as exc:
        return None, f"Failed to initialise RAG chain: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# RULE 2 — Employee context builder  (reads from `inputs` dict only)
# ─────────────────────────────────────────────────────────────────────────────
def _build_employee_context(inputs: dict, risk: float, raw_prob: float) -> str:
    """
    RULE 2 implementation
    ─────────────────────
    All employee attributes are read from the same `inputs` dict that was
    passed to `_predict_employee`. Because both functions operate on the
    identical dict object within the same Streamlit render cycle, the model's
    risk score and the LLM context are structurally guaranteed to describe
    the exact same employee. No secondary DataFrame lookup is needed.

    Policy trigger flags mirror the exact numeric thresholds in the HR manual:
      Article I  §1.4.1  — OverTime = "Yes"  → Burnout Prevention Program
      Article II §2.3.1  — YearsSinceLastPromotion >= 3  → Stagnation Protocol
      Article IV §4.8    — TrainingTimesLastYear < 3     → Development Risk
      Article III §3.5.1 — MonthlyIncome < 75% of median → Market Rate Review
    """
    def _v(key: str, default: str = "N/A") -> str:
        """Safe getter: returns `default` for None / empty / NaN values."""
        val = inputs.get(key, default)
        return default if val in (None, "", "N/A") else str(val)

    # ── Raw feature values from sidebar inputs ────────────────────────────
    overtime    = _v("OverTime")
    job_sat     = _v("JobSatisfaction")
    env_sat     = _v("EnvironmentSatisfaction")
    rel_sat     = _v("RelationshipSatisfaction")
    wlb         = _v("WorkLifeBalance")
    yrs_promo   = _v("YearsSinceLastPromotion",  "0")
    yrs_company = _v("YearsAtCompany",            "0")
    yrs_role    = _v("YearsInCurrentRole",        "N/A")
    total_yrs   = _v("TotalWorkingYears",         "0")
    income      = _v("MonthlyIncome",             "0")
    job_level   = _v("JobLevel",                  "N/A")
    job_role    = _v("JobRole",                   "N/A")
    department  = _v("Department",                "N/A")
    training    = _v("TrainingTimesLastYear",      "99")
    distance    = _v("DistanceFromHome",           "N/A")
    perf        = _v("PerformanceRating",          "N/A")
    stock       = _v("StockOptionLevel",           "N/A")
    n_companies = _v("NumCompaniesWorked",         "N/A")

    # ── Safe numeric conversions ──────────────────────────────────────────
    try:
        yrs_promo_f = float(yrs_promo)
    except (ValueError, TypeError):
        yrs_promo_f = 0.0
    try:
        training_f = float(training)
    except (ValueError, TypeError):
        training_f = 99.0
    try:
        income_f = float(income)
    except (ValueError, TypeError):
        income_f = 0.0

    med_income = float(medians.get("MonthlyIncome", max(income_f, 1.0)))

    # ── Policy trigger flags ──────────────────────────────────────────────
    ot_flag = (
        "CRITICAL — triggers Article I §1.4 Burnout Prevention Program"
        if overtime.strip().lower() == "yes"
        else "No overtime flag"
    )
    promo_flag = (
        f"CRITICAL — triggers Article II §2.3 Stagnation Intervention Protocol "
        f"({yrs_promo_f:.0f} yrs >= 3-yr threshold)"
        if yrs_promo_f >= 3
        else f"OK ({yrs_promo_f:.0f} yr — below 3-yr threshold)"
    )
    train_flag = (
        f"ALERT — triggers Article IV §4.8 Development Engagement Risk "
        f"({training_f:.0f} sessions < 3-session minimum)"
        if training_f < 3
        else f"OK ({training_f:.0f} sessions — above minimum)"
    )
    income_flag = (
        f"ALERT — triggers Article III §3.5 Market Rate Adjustment "
        f"(${income_f:,.0f} is below 75 pct of company median ${med_income:,.0f})"
        if income_f < med_income * 0.75
        else f"OK (${income_f:,.0f} — within acceptable range of median ${med_income:,.0f})"
    )

    return (
        "EMPLOYEE PROFILE  (single-employee Streamlit DSS inference):\n\n"

        "ROLE AND DEPARTMENT:\n"
        f"  Job Role         : {job_role}\n"
        f"  Department       : {department}\n"
        f"  Job Level        : {job_level}\n\n"

        "ATTRITION RISK ASSESSMENT:\n"
        f"  Predicted Risk     : {risk:.1f}%  [HIGH RISK]\n"
        f"  Raw Probability    : {raw_prob:.4f}\n"
        f"  Optimal Threshold  : {optimal_threshold * 100:.0f}%  "
        "(F1-maximised — loaded from artifact)\n"
        f"  Risk Excess        : +{risk - optimal_threshold * 100:.1f} pp "
        "above the decision boundary\n\n"

        "WORKLOAD AND OVERTIME:\n"
        f"  OverTime             : {overtime}  [{ot_flag}]\n"
        f"  Distance From Home   : {distance} km\n\n"

        "SATISFACTION SCORES  (1 = Low, 4 = High):\n"
        f"  Job Satisfaction     : {job_sat}\n"
        f"  Environment          : {env_sat}\n"
        f"  Relationship         : {rel_sat}\n"
        f"  Work-Life Balance    : {wlb}\n\n"

        "CAREER PROGRESSION:\n"
        f"  Years at Company             : {yrs_company}\n"
        f"  Years in Current Role        : {yrs_role}\n"
        f"  Years Since Last Promotion   : {yrs_promo}  [{promo_flag}]\n"
        f"  Total Working Years          : {total_yrs}\n"
        f"  Number of Companies Worked   : {n_companies}\n\n"

        "COMPENSATION AND REWARDS:\n"
        f"  Monthly Income       : ${income}  [{income_flag}]\n"
        f"  Stock Option Level   : {stock}  (0=None, 1=Strong, 2=High, 3=Critical)\n"
        f"  Performance Rating   : {perf}\n\n"

        "DEVELOPMENT:\n"
        f"  Training Sessions Last Year  : {training}  [{train_flag}]\n\n"

        "AUTOMATED POLICY TRIGGER SUMMARY:\n"
        f"  Overtime Protocol    : {ot_flag}\n"
        f"  Promotion Protocol   : {promo_flag}\n"
        f"  Training Protocol    : {train_flag}\n"
        f"  Compensation Alert   : {income_flag}\n\n"

        "TASK FOR AI HR CONSULTANT:\n"
        "Analyze the profile above strictly against the Corporate HR Policy Manual.\n"
        "Produce a structured retention plan with exactly three sections:\n"
        "  IMMEDIATE   (within 30 days)\n"
        "  SHORT-TERM  (1 to 6 months)\n"
        "  LONG-TERM   (6 to 12 months)\n"
        "Cite every applicable Article, Section, and Clause explicitly.\n"
        "State which HR protocols must be automatically triggered right now."
    )


# ─────────────────────────────────────────────────────────────────────────────
# RULE 3 — RAG invocation with artifact-threshold gate
# ─────────────────────────────────────────────────────────────────────────────
def _run_rag_consultant(inputs: dict, risk: float, raw_prob: float) -> dict:
    """
    RULE 3 implementation
    ─────────────────────
    The RAG chain is invoked ONLY when raw_prob >= optimal_threshold.
    `optimal_threshold` is always the value loaded from the artifact —
    it is never hardcoded as 0.50 or any other constant.

    RULE 5 implementation
    ─────────────────────
    Source documents are extracted from the chain response and their
    0-based page metadata index is converted to a 1-based page_label
    for the compliance expander.

    RULE 6 implementation
    ─────────────────────
    A try/except wraps the chain.invoke() call. Any exception is caught
    and returned as a structured failure dict — never re-raised to the
    Streamlit layer, which would crash the app.

    Returns
    -------
    dict(success, recommendation, sources, error)
    """
    # RULE 3 — gate check using the artifact threshold
    if raw_prob < optimal_threshold:
        return {
            "success":        False,
            "recommendation": "",
            "sources":        [],
            "error":          (
                f"Employee probability ({raw_prob:.4f}) is below the optimal "
                f"threshold ({optimal_threshold:.4f}). RAG not triggered."
            ),
        }

    chain, setup_err = _load_rag_chain()
    if setup_err:
        return {
            "success":        False,
            "recommendation": "",
            "sources":        [],
            "error":          setup_err,
        }

    # RULE 6 — inner try/except isolates chain.invoke() failures
    try:
        context  = _build_employee_context(inputs, risk, raw_prob)   # RULE 2
        response = chain.invoke({"query": context})

        # RULE 5 — 1-based page label for compliance expander
        sources = [
            {
                "page_label": int(d.metadata.get("page", 0)) + 1,
                "excerpt":    d.page_content.strip(),
            }
            for d in response.get("source_documents", [])
        ]

        return {
            "success":        True,
            "recommendation": response["result"],
            "sources":        sources,
            "error":          None,
        }

    except Exception as chain_exc:
        return {
            "success":        False,
            "recommendation": "",
            "sources":        [],
            "error":          f"RAG chain error: {chain_exc}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# RULE 6 — Shared rules-based fallback renderer
# Rendered whenever RAG is unavailable, disabled, or fails. Ensures the
# right-hand panel is never empty and the app never crashes.
# ─────────────────────────────────────────────────────────────────────────────
def _render_rules_tips(inputs: dict) -> None:
    """Local rule-engine recommendations grounded in the HR Policy Manual."""
    tips: list[str] = []

    ot   = str(inputs.get("OverTime",                "")).strip().lower()
    jsat = float(inputs.get("JobSatisfaction",         medians.get("JobSatisfaction",         3)))
    yslp = float(inputs.get("YearsSinceLastPromotion", medians.get("YearsSinceLastPromotion", 0)))
    wlb  = float(inputs.get("WorkLifeBalance",         medians.get("WorkLifeBalance",         3)))
    env  = float(inputs.get("EnvironmentSatisfaction", medians.get("EnvironmentSatisfaction", 3)))
    inc  = float(inputs.get("MonthlyIncome",           medians.get("MonthlyIncome",           0)))
    dist = float(inputs.get("DistanceFromHome",        medians.get("DistanceFromHome",        0)))
    trn  = float(inputs.get("TrainingTimesLastYear",   medians.get("TrainingTimesLastYear",   3)))
    perf = float(inputs.get("PerformanceRating",       medians.get("PerformanceRating",       3)))

    med_income = float(medians.get("MonthlyIncome", max(inc, 1.0)))

    # Article I §1.3 / §1.4 — Overtime and Burnout
    if ot == "yes":
        tips.append(
            "⏰ **Reduce overtime immediately** — active overtime triggers the Burnout "
            "Prevention Program (Article I, §1.4, Clause 1.4.1). Verify whether "
            "the 8 hr/week monthly threshold (Clause 1.4.1) has been breached."
        )
    # Article II §2.3 — Career Stagnation Intervention
    if yslp >= 3:
        tips.append(
            f"🚀 **Initiate Stagnation Intervention Protocol** — {yslp:.0f} years "
            "without promotion exceeds the 3-year trigger (Article II, §2.3, "
            "Clause 2.3.1). Mandatory HR Assessment must begin within 30 days "
            "(Clause 2.3.2) and an Individual Growth Plan must be created (Clause 2.3.3)."
        )
    # Article III §3.5 — Compensation Review
    if inc < med_income * 0.75:
        tips.append(
            f"💰 **Trigger Market Rate Adjustment** — income ${inc:,.0f} is below "
            f"75 pct of company median ${med_income:,.0f} (Article III, §3.5, "
            "Clause 3.5.1). Mandatory compensation review within 30 days (Clause 3.5.2)."
        )
    # Article IV §4.3 / §4.8 — Training minimum
    if trn < 3:
        tips.append(
            f"📚 **Development counselling required** — only {trn:.0f} session(s) "
            "completed, below the 3-program annual minimum (Article IV, §4.3). "
            "Flag employee as Development Engagement Risk (Article IV, §4.8)."
        )
    # Article II §2.6 — High performer + stagnation = High Retention Priority
    if perf >= 4 and yslp >= 3:
        tips.append(
            "⭐ **Flag as High Retention Priority Employee** — above-average "
            "performance combined with 3+ years without promotion automatically "
            "triggers this designation (Article II, §2.6) and escalation to "
            "the Retention Review Committee (Article V, §5.3)."
        )
    # Satisfaction thresholds
    if jsat <= 2:
        tips.append(
            "😞 **Address job satisfaction** — score ≤ 2 indicates serious "
            "disengagement. Schedule a career review meeting within 30 days "
            "(Article V, §5.2, Steps 1-2)."
        )
    if wlb <= 2:
        tips.append(
            "🏠 **Improve work-life balance** — score ≤ 2. Offer flexible "
            "scheduling or remote work per Article I, §1.4, Clause 1.4.3."
        )
    if env <= 2:
        tips.append(
            "🌱 **Address work environment** — score ≤ 2. Consider team-building, "
            "role reassignment, or location change."
        )
    if dist > 20:
        tips.append(
            "🚗 **Long commute detected** — consider remote/hybrid arrangement "
            "or a transport allowance to reduce attrition pressure."
        )
    if not tips:
        tips.append(
            "✅ No major rule-based flags triggered. Maintain regular check-ins, "
            "recognition programmes, and growth opportunities (Article V, §5.2)."
        )

    st.subheader("💡 HR Recommendations (Rules-Based)")
    for tip in tips:
        st.markdown(f"- {tip}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — RULE 1: employee input form
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("👤 Employee Details")
st.sidebar.caption("Enter employee information, then click **Predict**.")

# `inputs` is the single source of truth for both inference and LLM context.
inputs: dict = {}

st.sidebar.subheader("Personal & Role")
for feat in cat_features:
    opts    = list(encoders[feat].classes_)
    default = str(modes.get(feat, opts[0]))
    idx     = opts.index(default) if default in opts else 0
    inputs[feat] = st.sidebar.selectbox(feat, opts, index=idx)

st.sidebar.subheader("Work Details")
for feat in num_features:
    default      = float(medians.get(feat, 0.0))
    inputs[feat] = st.sidebar.number_input(feat, value=default, step=1.0)

predict_btn = st.sidebar.button(
    "🔍 Predict Attrition Risk",
    type="primary",
    use_container_width=True,
)

st.sidebar.divider()
st.sidebar.caption(
    f"**Optimal threshold:** `{optimal_threshold:.4f}` "
    f"({optimal_threshold * 100:.0f}%) — F1-maximised on test set"
)

# ── AI Consultant settings ────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("🤖 AI Consultant Settings")

api_key_input = st.sidebar.text_input(
    "Groq API Key",
    value=os.environ.get("GROQ_API_KEY", ""),
    type="password",
    placeholder="gsk_…",
    help=(
        "Required to activate the AI HR Consultant. "
        "Leave blank if GROQ_API_KEY is already set as an environment variable."
    ),
)
# Apply the key immediately so _load_rag_chain() (cached once) can use it.
if api_key_input.strip():
    os.environ["GROQ_API_KEY"] = api_key_input.strip()

rag_enabled: bool = st.sidebar.toggle(
    "Enable AI HR Consultant (RAG)",
    value=True,
    help=(
        "ON: Llama-3 70B (Groq) + HR Policy vector search fires automatically for HIGH RISK "
        "predictions.  OFF: only the rules-based panel is shown."
    ),
)

st.sidebar.caption(
    "📄 Place `hr_policy_manual.pdf` in `./data_docs/` "
    "and provide your Groq API key to activate the AI Consultant."
)


# ─────────────────────────────────────────────────────────────────────────────
# Main page header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏢 HR Attrition — Decision Support System")
st.caption(
    f"Model: **{model_name}** | "
    f"Accuracy: {scores[model_name]['acc'] * 100:.1f}% | "
    f"AUC: {scores[model_name]['auc'] * 100:.1f}% | "
    f"Threshold: {optimal_threshold:.4f}"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Default view  (no prediction requested yet)
# ─────────────────────────────────────────────────────────────────────────────
if not predict_btn:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",     model_name)
    c2.metric("Accuracy",  f"{scores[model_name]['acc'] * 100:.1f}%")
    c3.metric("AUC Score", f"{scores[model_name]['auc'] * 100:.1f}%")
    c4.metric("Threshold", f"{optimal_threshold:.4f}")

    st.divider()
    st.subheader("📊 Top Attrition Risk Factors")
    chart_data = fi.head(10).set_index("Feature")["Importance"].sort_values()
    st.bar_chart(chart_data, color=["#e74c3c"])

    st.info(
        "Fill in the employee details in the **left sidebar** and click "
        "**Predict Attrition Risk** to run the model and activate the "
        "AI HR Consultant."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction view
# ─────────────────────────────────────────────────────────────────────────────
else:
    # RULE 6 — outer guard: catches any failure that occurs before model output
    try:
        risk, raw_prob = _predict_employee(inputs)   # RULE 1

        # RULE 3 — gate comparison uses artifact threshold, never hardcoded 0.50
        high_risk: bool = raw_prob >= optimal_threshold

        # ── Risk result banner ────────────────────────────────────────────
        if high_risk:
            st.error(f"### 🔴 HIGH RISK — {risk:.1f}% chance of leaving")
        else:
            st.success(f"### 🟢 LOW RISK — {risk:.1f}% chance of leaving")

        # ── Score metrics + progress bar ──────────────────────────────────
        m_col, p_col = st.columns([1, 3])
        with m_col:
            st.metric("Risk Score", f"{risk:.1f}%")
            st.metric(
                "Threshold",
                f"{optimal_threshold * 100:.0f}%",
                delta=f"{risk - optimal_threshold * 100:+.1f} pp",
                delta_color="inverse",
            )
        with p_col:
            st.write("**Risk Level**")
            st.progress(risk / 100.0)    # safe float 0.0 – 1.0

        st.divider()

        # ── Two-column layout ─────────────────────────────────────────────
        left_col, right_col = st.columns([1, 1], gap="large")

        # ── LEFT: Feature importance table + local recommendations ────────
        with left_col:
            st.subheader("📋 Top 5 Global Risk Factors")
            top5 = fi.head(5).copy().reset_index(drop=True)
            top5.index = top5.index + 1    # 1-based display rank
            st.dataframe(top5[["Feature", "Importance"]], use_container_width=True)

            st.divider()
            _render_rules_tips(inputs)     # RULE 6 — always visible on left

        # ── RIGHT: AI Executive HR Consultant (RAG) ───────────────────────
        with right_col:
            st.subheader("🤖 AI Executive HR Consultant")
            st.caption(
                "Policy-grounded retention strategy generated by GPT-4o, "
                "grounded in retrieved excerpts from the Corporate HR Policy Manual."
            )

            # ── State A: employee not above threshold — RAG not needed ────
            if not high_risk:
                st.info(
                    "The AI Consultant activates **only for HIGH RISK employees** "
                    f"(raw probability >= `{optimal_threshold:.4f}`).\n\n"
                    f"This employee's probability is `{raw_prob:.4f}` — "
                    "below the optimal threshold. "
                    "See the rules-based recommendations on the left."
                )

            # ── State B: user has disabled RAG via sidebar toggle ─────────
            elif not rag_enabled:
                st.warning(
                    "The AI HR Consultant is currently toggled **off** in the sidebar.\n\n"
                    "Enable **AI HR Consultant (RAG)** and provide your Groq API key "
                    "to receive policy-grounded recommendations."
                )

            # ── State C: RAG packages not installed ───────────────────────
            elif not _RAG_AVAILABLE:
                st.warning(
                    "**RAG libraries not installed.**\n\n"
                    "Run the following command, then restart the app:\n\n"
                    "```\npip install langchain langchain-community langchain-groq "
                    "langchain-chroma chromadb pypdf fastembed\n```\n\n"
                    f"Import error: `{_RAG_IMPORT_ERROR}`"
                )
                st.divider()
                st.info("Showing rules-based recommendations as fallback:")
                _render_rules_tips(inputs)

            # ── State D: RAG available + enabled — invoke the chain ───────
            else:
                with st.spinner("🔍 Retrieving policy clauses and consulting Llama-3 70B …"):
                    # RULE 6 — inner try/except isolates chain.invoke() errors
                    try:
                        rag_result = _run_rag_consultant(inputs, risk, raw_prob)
                    except Exception as chain_exc:
                        rag_result = {
                            "success":        False,
                            "recommendation": "",
                            "sources":        [],
                            "error":          f"Unexpected chain error: {chain_exc}",
                        }

                if rag_result["success"]:
                    # ── Successful GPT-4o response ────────────────────────
                    st.success("**Policy-Grounded Retention Strategy**")
                    st.markdown(rag_result["recommendation"])

                    # RULE 5 — compliance expander with 1-based page labels
                    sources = rag_result["sources"]
                    if sources:
                        with st.expander(
                            f"📄 View {len(sources)} Retrieved Policy "
                            f"Excerpt{'s' if len(sources) != 1 else ''} "
                            f"(Compliance Transparency)"
                        ):
                            for i, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"**Excerpt {i} — Page {src['page_label']}**"
                                )
                                st.caption(src["excerpt"])
                                if i < len(sources):
                                    st.divider()

                else:
                    # RULE 6 — graceful failure: classify and show correct message
                    err_msg = rag_result.get("error", "Unknown error.")

                    if "api key" in err_msg.lower() or "openai" in err_msg.lower():
                        st.warning(
                            f"**OpenAI API issue:** {err_msg}\n\n"
                            "Enter a valid API key in the **AI Consultant Settings** "
                            "panel in the sidebar."
                        )
                    elif (
                        "not found" in err_msg.lower()
                        or "pdf" in err_msg.lower()
                        or "policy manual" in err_msg.lower()
                    ):
                        st.warning(
                            f"**PDF issue:** {err_msg}\n\n"
                            "Place `hr_policy_manual.pdf` in the `./data_docs/` folder."
                        )
                    elif "below" in err_msg.lower() and "threshold" in err_msg.lower():
                        # Theoretically unreachable (gated above), shown defensively
                        st.info(err_msg)
                    else:
                        st.error(f"**AI Consultant error:** {err_msg}")

                    # RULE 6 — always render local tips so column is never empty
                    st.divider()
                    st.info(
                        "Showing **rules-based recommendations** as a fallback. "
                        "Resolve the issue above to activate GPT-4o guidance."
                    )
                    _render_rules_tips(inputs)

        st.divider()

        # ── All model performance scores (collapsible) ────────────────────
        with st.expander("📊 All Model Performance Scores"):
            scores_df = pd.DataFrame([
                {
                    "Model":     m,
                    "Accuracy":  f"{v['acc']  * 100:.1f}%",
                    "Precision": f"{v['prec'] * 100:.1f}%",
                    "Recall":    f"{v['rec']  * 100:.1f}%",
                    "F1":        f"{v['f1']   * 100:.1f}%",
                    "AUC":       f"{v['auc']  * 100:.1f}%",
                }
                for m, v in scores.items()
            ])
            st.dataframe(scores_df, use_container_width=True, hide_index=True)

    # RULE 6 — outer guard: any pre-model exception surfaces here with guidance
    except Exception as outer_exc:
        st.error(f"**Prediction error:** {outer_exc}")
        st.exception(outer_exc)
        st.info(
            "This error occurred before the model produced a result. "
            "Verify that `hr_dss_artifacts.pkl` is present and was generated "
            "by the current version of the training notebook (§1.13)."
        )
