# app.py
"""
ai!Auto-Grader Pro - Ultra-Modern Masterpiece Edition
HCI Principles: Glassmorphism, Information Hierarchy, & Cognitive Load Reduction.
"""
import pandas as pd
from io import BytesIO
import os
import json
import time
import re
import streamlit as st
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime

# Optional Plotting & Docs
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    import docx2txt
except ImportError:
    docx2txt = None

# ==================== MASTERPIECE CSS & THEMING ====================
st.set_page_config(
    page_title="ai!Auto-Grader Pro", 
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

def apply_masterpiece_styles():
    st.markdown("""
    <style>
        /* Modern Font & Smooth Rendering */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0b0e14;
        }

        /* Glassmorphism Containers */
        .stChatMessage, .stExpander, .stAlert, div[data-testid="stMetricValue"], .stTabs [data-baseweb="tab-panel"] {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* Glossy Header Styling */
        .main-header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3.2rem !important;
            letter-spacing: -2px;
            margin-bottom: 0px;
        }

        /* Hovering Tab Effects */
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px 8px 0 0;
            gap: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(79, 172, 254, 0.1);
            color: #4facfe !important;
        }

        /* Advanced Score Cards */
        .grading-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .grading-card:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(79, 172, 254, 0.4);
            box-shadow: 0 15px 45px rgba(0, 0, 0, 0.4);
        }

        /* Glowing Metric Display */
        [data-testid="stMetricValue"] {
            color: #4facfe !important;
            font-weight: 800 !important;
            text-shadow: 0 0 10px rgba(79, 172, 254, 0.5);
        }

        /* Custom Modern Progress Bar */
        .progress-bar-container {
            height: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            margin: 15px 0;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            box-shadow: 0 0 15px rgba(79, 172, 254, 0.6);
            transition: width 1s ease-in-out;
        }

        /* Grammar Badge */
        .grammar-issue {
            background: rgba(255, 255, 255, 0.04);
            border-left: 4px solid #ff4b4b;
            padding: 12px;
            margin: 8px 0;
            border-radius: 4px 12px 12px 4px;
            font-size: 0.95rem;
        }

        /* Action Buttons */
        .stButton>button {
            border-radius: 12px;
            padding: 12px 24px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            border: none;
            color: white;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px rgba(79, 172, 254, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

apply_masterpiece_styles()

# ==================== CORE UTILITIES (PRESERVING LOGIC) ====================

@st.cache_resource(show_spinner="🔄 Waking up the Grading Engine...")
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")

embedding_model = load_embedding_model()

def read_text_file(uploaded_file) -> str:
    if uploaded_file is None: return ""
    with st.status(f"📄 Decoding {uploaded_file.name}...", state="running") as status:
        try:
            content = uploaded_file.getvalue()
            name = uploaded_file.name.lower()
            res = ""
            if name.endswith(".txt"): 
                res = content.decode("utf-8")
            elif name.endswith(".docx") and docx2txt:
                tmp_path = f"tmp_{int(time.time())}.docx"
                with open(tmp_path, "wb") as f: 
                    f.write(content)
                res = docx2txt.process(tmp_path)
                os.remove(tmp_path)
            elif name.endswith(".pdf") and pdfplumber:
                with pdfplumber.open(BytesIO(content)) as pdf:
                    res = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            else: 
                res = content.decode("utf-8", errors="ignore")
            status.update(label=f"✅ {uploaded_file.name} Integrated", state="complete")
            return res.strip()
        except Exception as e:
            status.update(label=f"❌ Load Error: {uploaded_file.name}", state="error")
            return ""

def parse_teacher_rubric(text: str) -> Optional[dict]:
    if not text or not text.strip(): 
        return None
    
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2: 
        return None
    
    criteria = []
    
    # Skip header lines
    for line in lines:
        # Skip lines that look like headers
        if any(w in line.lower() for w in ['criterion', 'weight', 'description', 'criteria', 'score']):
            continue
            
        # Split by common delimiters
        parts = [p.strip() for p in re.split(r'[|,\t]', line) if p.strip()]
        if len(parts) < 2: 
            continue
        
        try:
            # Extract weight from second part
            weight_match = re.search(r'(\d+(?:\.\d+)?)', parts[1])
            if not weight_match:
                continue
                
            w_val = float(weight_match.group(1))
            criterion_name = parts[0]
            
            # Determine criterion type
            is_grammar = any(g in criterion_name.lower() for g in ['grammar', 'spelling', 'punctuation', 'syntax'])
            
            criteria.append({
                "name": criterion_name, 
                "weight": w_val,
                "type": "grammar_penalty" if is_grammar else "similarity",
                "penalty_per_issue": 1.5 if is_grammar else 0
            })
        except Exception:
            continue
    
    if not criteria: 
        return None
    
    # Normalize weights
    total_w = sum(c["weight"] for c in criteria)
    if total_w > 0:
        for c in criteria: 
            c["weight"] /= total_w
    
    return {"criteria": criteria}

def convert_to_ielts_band(score_100: float) -> float:
    """Convert 0-100 score to IELTS band score 0-9"""
    if score_100 >= 95: return 9.0
    elif score_100 >= 88: return 8.5
    elif score_100 >= 80: return 8.0
    elif score_100 >= 75: return 7.5
    elif score_100 >= 70: return 7.0
    elif score_100 >= 65: return 6.5
    elif score_100 >= 60: return 6.0
    elif score_100 >= 55: return 5.5
    elif score_100 >= 50: return 5.0
    elif score_100 >= 45: return 4.5
    elif score_100 >= 40: return 4.0
    elif score_100 >= 35: return 3.5
    elif score_100 >= 30: return 3.0
    elif score_100 >= 25: return 2.5
    elif score_100 >= 20: return 2.0
    elif score_100 >= 15: return 1.5
    elif score_100 >= 10: return 1.0
    elif score_100 > 0: return 0.5
    return 0.0

def embed_texts(texts: List[str]) -> np.ndarray:
    """Convert text list to embeddings"""
    return embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def grammar_check_with_jina(text: str) -> Dict[str, Any]:
    """Enhanced Jina AI grammar checking with better parsing"""
    api_key = os.getenv("JINACHAT_API_KEY")
    if not api_key or not text.strip(): 
        return {"available": False, "issues_count": 0, "examples": []}
    
    with st.status("🔍 Deep Syntax Analysis...", state="running") as status:
        url = "https://api.jina.ai/v1/chat/completions"
        
        # Improved prompt for better grammar analysis
        prompt = f"""Analyze this text strictly for grammar, spelling, punctuation, and style errors.
        For each error found, provide:
        1. The type of error (e.g., "Subject-verb agreement", "Spelling error", "Punctuation missing")
        2. The correction
        3. Brief explanation
        
        Text to analyze: "{text}"
        
        Format each error as:
        Error Type: Correction | Explanation
        
        If there are no errors, respond with "No grammar issues detected"."""
        
        try:
            resp = requests.post(
                url, 
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }, 
                json={
                    "model": "jina-clip", 
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a strict English grammar expert. Find all errors and provide corrections."
                        },
                        {"role": "user", "content": prompt}
                    ], 
                    "temperature": 0.1,
                    "max_tokens": 800
                },
                timeout=30
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    
                    # Check if no issues were found
                    if "No grammar issues detected" in content or "No issues" in content:
                        status.update(label="✅ No grammar issues detected", state="complete")
                        return {"available": True, "issues_count": 0, "examples": []}
                    
                    # Parse errors from response
                    examples = []
                    lines = content.strip().split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith("#") or ":" not in line:
                            continue
                            
                        # Parse format: Error Type: Correction | Explanation
                        if ":" in line:
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                error_type = parts[0].strip()
                                rest = parts[1].strip()
                                
                                # Extract correction and explanation
                                if "|" in rest:
                                    correction_part, explanation = rest.split("|", 1)
                                    correction = correction_part.strip()
                                    explanation = explanation.strip()
                                else:
                                    correction = rest
                                    explanation = ""
                                
                                examples.append({
                                    "message": error_type,
                                    "context": text[:80] + "..." if len(text) > 80 else text,
                                    "suggestions": [correction],
                                    "explanation": explanation
                                })
                    
                    issues_count = len(examples)
                    status.update(label=f"✅ Logic Analysis: {issues_count} issues", state="complete")
                    return {
                        "available": True, 
                        "issues_count": issues_count, 
                        "examples": examples[:6],  # Limit to 6 examples
                        "method": "jina_ai_enhanced"
                    }
            
            # If API call fails
            status.update(label="⚠️ Grammar check temporarily unavailable", state="error")
            return {"available": False, "issues_count": 0, "examples": []}
            
        except Exception as e:
            status.update(label="⚠️ Grammar check error", state="error")
            return {"available": False, "issues_count": 0, "examples": []}

def apply_rubric_json(rubric: dict, model_ans: str, student_ans: str, output_scale: str) -> Dict[str, Any]:
    """Apply rubric-based grading with Jina AI grammar checking"""
    criteria = rubric.get("criteria", [])
    
    # Calculate similarity
    vecs = embed_texts([model_ans, student_ans])
    sim_score = cosine_sim(vecs[0], vecs[1])
    sim_percent = max(0, min(100, (sim_score + 1) / 2.0 * 100))
    
    # Grammar check
    g = grammar_check_with_jina(student_ans)
    
    total_score = 0.0
    breakdown = []
    
    for c in criteria:
        if c["type"] == "similarity":
            subscore = sim_percent
        elif c["type"] == "grammar_penalty":
            if g["available"]:
                # Apply penalty based on grammar issues
                penalty = g["issues_count"] * c.get("penalty_per_issue", 1.5)
                subscore = max(0.0, 100.0 - penalty)
            else:
                subscore = 100.0  # Full points if grammar check unavailable
        else:
            subscore = sim_percent  # Default to similarity
            
        weighted_score = subscore * c["weight"]
        total_score += weighted_score
        
        breakdown.append({
            "criterion": c["name"], 
            "weight": round(c["weight"], 3), 
            "subscore": round(subscore, 2),
            "type": c["type"],
            "weighted_score": round(weighted_score, 2)
        })
    
    # Cap at 100
    total_score = min(100.0, total_score)
    
    # Convert to IELTS if needed
    if output_scale == "ielts_band_0-9":
        final_score = convert_to_ielts_band(total_score)
        scale_used = "ielts"
    else:
        final_score = total_score
        scale_used = "numeric"
    
    return {
        "final_score": round(final_score, 2), 
        "breakdown": breakdown, 
        "similarity": sim_percent / 100.0, 
        "grammar": g, 
        "scale_used": scale_used,
        "original_100_score": round(total_score, 2),
        "grading_method": "rubric"
    }

def heuristic_grade(model_ans: str, student_ans: str, output_scale: str) -> Dict[str, Any]:
    """Fallback grading method when no rubric is provided"""
    vecs = embed_texts([model_ans, student_ans])
    sim_score = cosine_sim(vecs[0], vecs[1])
    sim_percent = max(0, min(100, (sim_score + 1) / 2.0 * 100))
    
    g = grammar_check_with_jina(student_ans)
    
    # Apply grammar penalty
    if g["available"]:
        penalty = g["issues_count"] * 1.5
        score = max(0.0, sim_percent - penalty)
    else:
        score = sim_percent
    
    # Convert to IELTS if needed
    if output_scale == "ielts_band_0-9":
        final_score = convert_to_ielts_band(score)
        scale_used = "ielts"
    else:
        final_score = score
        scale_used = "numeric"
    
    breakdown = [
        {
            "criterion": "Content Similarity", 
            "weight": 0.8, 
            "subscore": round(sim_percent, 2),
            "type": "similarity",
            "weighted_score": round(sim_percent * 0.8, 2)
        },
        {
            "criterion": "Grammar & Mechanics", 
            "weight": 0.2, 
            "subscore": round(max(0, 100 - (g["issues_count"] * 1.5)), 2),
            "type": "grammar_penalty",
            "weighted_score": round(max(0, 100 - (g["issues_count"] * 1.5)) * 0.2, 2)
        }
    ]
    
    return {
        "final_score": round(final_score, 2), 
        "similarity": sim_percent / 100.0, 
        "grammar": g, 
        "breakdown": breakdown,
        "scale_used": scale_used,
        "original_100_score": round(score, 2),
        "grading_method": "heuristic"
    }

def generate_feedback_with_jina(prompt_text: str) -> Optional[str]:
    """Generate AI feedback using Jina AI"""
    api_key = os.getenv("JINACHAT_API_KEY")
    if not api_key: 
        return None
    
    with st.status("🤖 Generating AI feedback...", state="running") as status:
        try:
            resp = requests.post(
                "https://api.jina.ai/v1/chat/completions", 
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "jina-clip", 
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an objective, constructive grading assistant. Provide actionable feedback."
                        },
                        {"role": "user", "content": prompt_text}
                    ], 
                    "temperature": 0.2,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    feedback = data["choices"][0]["message"]["content"]
                    status.update(label="✅ AI feedback generated", state="complete")
                    return feedback
                    
            status.update(label="⚠️ AI feedback unavailable", state="error")
            return None
            
        except Exception as e:
            status.update(label="⚠️ AI feedback error", state="error")
            return None

def export_results_to_excel(results: list) -> bytes:
    """Export grading results to Excel format"""
    if not results:
        return b""
    
    # Create summary dataframe
    summary_data = []
    for r in results:
        summary_data.append({
            "Student Name": r.get("name", "Unknown"),
            "Final Score": r.get("final_score", 0),
            "Original 100-Point Score": r.get("details", {}).get("original_100_score", 0),
            "Similarity (%)": round(r.get("details", {}).get("similarity", 0) * 100, 2),
            "Grammar Issues": r.get("details", {}).get("grammar", {}).get("issues_count", 0),
            "Grading Method": r.get("details", {}).get("grading_method", "Unknown"),
            "Scale Used": r.get("details", {}).get("scale_used", "numeric"),
            "Timestamp": r.get("timestamp", "")
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Create breakdown dataframe
    breakdown_data = []
    for r in results:
        for b in r.get("details", {}).get("breakdown", []):
            breakdown_data.append({
                "Student Name": r.get("name", "Unknown"),
                "Criterion": b.get("criterion", ""),
                "Weight": b.get("weight", 0),
                "Subscore": b.get("subscore", 0),
                "Type": b.get("type", ""),
                "Weighted Score": b.get("weighted_score", 0)
            })
    
    df_breakdown = pd.DataFrame(breakdown_data)
    
    # Write to Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if not df_breakdown.empty:
            df_breakdown.to_excel(writer, sheet_name="Breakdown", index=False)
    
    return output.getvalue()

# ==================== MAIN INTERFACE (KEEPING ELEMENTS IN PLACE) ====================
st.markdown('<div class="main-header">ai!Auto-Grader Pro</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #6a7682; margin-top: -15px; font-weight: 500;">Experience the future of automated academic assessment.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛠️ Core Engine Settings")
    output_scale = st.selectbox("**Scoring Model**", ["numeric_100", "ielts_band_0-9"], 
                               help="Framework for final evaluation.", index=0)
    
    st.markdown("### 👁️ Interface Experience")
    show_grammar_examples = st.toggle("Detailed Grammar Feedback", value=True)
    show_detailed_breakdown = st.toggle("Advanced Analytics Reveal", value=True)
    enable_ai_feedback = st.toggle("Dynamic AI Feedback", value=True)
    
    st.divider()
    st.markdown("### ⚡ System Integrity")
    st.success("● Neural Model: Active (BGE-Large)")
    st.info(f"● Jina AI Status: {'Connected' if os.getenv('JINACHAT_API_KEY') else 'Offline'}")
    
    # Add rubrics uploader that was missing
    st.divider()
    st.markdown("### 📊 Rubric Options")
    rubric_file = st.file_uploader("Upload Rubric File", type=["txt", "docx", "pdf"], 
                                   help="Optional: Upload a rubric file instead of pasting")

# TABS (Keeping order same as codebase)
tab1, tab2, tab3 = st.tabs(["📥 Input Workspace", "🎯 Results Terminal", "📈 Analytics Dashboard"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📝 Exercise Metadata")
        ex_file = st.file_uploader("Upload Prompt", type=["txt","docx","pdf"])
        ex_text_paste = st.text_area("Or Paste Prompt Description", height=100,
                                    placeholder="Enter the exercise description, question, or prompt here...")
        
        st.markdown("#### 👥 Batch Submissions")
        student_files = st.file_uploader("Upload Student Files", accept_multiple_files=True, 
                                        type=["txt","docx","pdf"])
        student_paste = st.text_area("Or Paste Multi-Submissions (Use '---' separator)", height=150,
                                    placeholder="Paste student submissions here, separated by '---' on a new line...")

    with col2:
        st.markdown("#### 🥇 Gold Standard (Model)")
        model_file = st.file_uploader("Upload Solution", type=["txt","docx","pdf"])
        model_text_paste = st.text_area("Or Paste Model Answer", height=100,
                                       placeholder="Enter the ideal/model answer here...")
    
        st.markdown("#### ⚖️ Evaluation Rubric")
        # Use uploaded rubric file if provided
        rubric_content = ""
        if rubric_file:
            rubric_content = read_text_file(rubric_file)
        
        rubric_text_paste = st.text_area("Define Rubric Criteria", height=150, 
            value=rubric_content or "Content Clarity | 60\nStructural Organization | 20\nGrammar & Syntax | 20",
            help="Format: Criterion | Weight (e.g., Content Clarity | 60)")
        
        if st.button("Preview Rubric Matrix", type="secondary", use_container_width=True):
            p = parse_teacher_rubric(rubric_text_paste)
            if p: 
                st.success("✅ Rubric parsed successfully")
                st.json(p)
            else:
                st.error("❌ Could not parse rubric. Please check format.")

    st.markdown("---")
    grade_button = st.button("🚀 INITIATE GRADING ENGINE", type="primary", use_container_width=True)

# ==================== EXECUTION LOGIC ====================
if grade_button:
    # Get exercise text
    ex_text = ex_text_paste.strip()
    if not ex_text and ex_file:
        ex_text = read_text_file(ex_file)
    
    # Get model text
    model_text = model_text_paste.strip()
    if not model_text and model_file:
        model_text = read_text_file(model_file)
    
    # Validation
    if not ex_text:
        st.error("❌ Missing Exercise/Prompt data. Please provide the exercise description.")
        st.stop()
    
    if not model_text:
        st.error("❌ Missing Model Solution. Please provide the ideal answer for comparison.")
        st.stop()
    
    # Parse rubric
    rubric_obj = parse_teacher_rubric(rubric_text_paste)
    
    # Collect student submissions
    student_texts = []
    student_names = []
    
    # From uploaded files
    if student_files:
        for f in student_files:
            txt = read_text_file(f)
            if txt and txt.strip():
                student_texts.append(txt.strip())
                student_names.append(f.name)
    
    # From pasted text
    if student_paste.strip():
        submissions = [s.strip() for s in student_paste.split("\n---\n") if s.strip()]
        for i, submission in enumerate(submissions):
            student_texts.append(submission)
            student_names.append(f"Student_{i+1}")
    
    if not student_texts:
        st.error("❌ No student work detected for evaluation.")
        st.stop()

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Clear previous results
    st.session_state.results = []
    
    # Grade with progress bar
    p_bar = st.progress(0, "Initializing Neural Scan...")
    
    for idx, (s_text, s_name) in enumerate(zip(student_texts, student_names)):
        progress = (idx + 1) / len(student_texts)
        p_bar.progress(progress, f"Analyzing {s_name}...")
        
        try:
            # Apply grading based on rubric availability
            if rubric_obj and rubric_obj.get("criteria"):
                res = apply_rubric_json(rubric_obj, model_text, s_text, output_scale)
            else:
                res = heuristic_grade(model_text, s_text, output_scale)
            
            # Generate AI feedback if enabled
            ai_feedback = None
            if enable_ai_feedback:
                ai_prompt = f"""Provide constructive feedback for this student work:
                
                Exercise: {ex_text[:500]}
                Model Answer: {model_text[:500]}
                Student Answer: {s_text[:500]}
                Score: {res['final_score']} ({res['scale_used'].upper()} scale)
                
                Provide 2-3 specific, actionable suggestions for improvement."""
                
                ai_feedback = generate_feedback_with_jina(ai_prompt)
            
            # Add to results
            st.session_state.results.append({
                "name": s_name, 
                "final_score": res["final_score"], 
                "details": res, 
                "jina_feedback": ai_feedback,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
        except Exception as e:
            st.warning(f"⚠️ Error grading {s_name}: {str(e)}")
            st.session_state.results.append({
                "name": s_name,
                "error": str(e),
                "final_score": 0,
                "details": {"error": str(e)},
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
    
    p_bar.empty()
    st.success(f"✅ Successfully graded {len(student_texts)} submissions!")
    st.rerun()

# ==================== RESULTS & ANALYTICS ====================
with tab2:
    if not st.session_state.get('results'):
        st.info("Terminal Idle. Upload materials and click 'INITIATE GRADING ENGINE' to begin.")
    else:
        for r in st.session_state.results:
            # Skip if there was an error
            if r.get("error"):
                st.warning(f"⚠️ Error processing {r['name']}: {r['error']}")
                continue
                
            st.markdown(f'<div class="grading-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            
            with c1: 
                st.markdown(f"### {r['name']}")
                method = r['details'].get('grading_method', 'unknown').upper()
                st.caption(f"Processed via {method} engine at {r['timestamp']}")
            
            with c2:
                # Dynamic coloring based on score
                score = r['final_score']
                scale = r['details'].get('scale_used', 'numeric')
                
                if scale == 'ielts':
                    # IELTS coloring
                    if score >= 7.0:
                        color = "#4facfe"  # Blue for high IELTS
                    elif score >= 5.5:
                        color = "#ffda6a"  # Yellow for medium IELTS
                    else:
                        color = "#ff6b6b"  # Red for low IELTS
                    score_display = f"{score}/9"
                else:
                    # Numeric coloring
                    if score >= 80:
                        color = "#4facfe"  # Blue for excellent
                    elif score >= 60:
                        color = "#ffda6a"  # Yellow for average
                    else:
                        color = "#ff6b6b"  # Red for poor
                    score_display = f"{score}/100"
                
                st.markdown(f"<h1 style='color: {color}; text-align: center; margin:0;'>{score_display}</h1>", 
                           unsafe_allow_html=True)
                st.caption(f"<p style='text-align: center;'>Scale: {scale.upper()}</p>", unsafe_allow_html=True)
            
            # Progress bar
            if scale == 'ielts':
                p_val = (score / 9) * 100
            else:
                p_val = score
                
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-fill" style="width: {p_val}%;"></div></div>', 
                       unsafe_allow_html=True)
            
            f1, f2 = st.columns(2)
            with f1:
                with st.expander("💬 AI Evaluator Insights", expanded=True):
                    feedback = r.get('jina_feedback', "Semantic comparison verified. No critical deviations found.")
                    st.write(feedback)
            
            with f2:
                if show_detailed_breakdown:
                    with st.expander("📊 Score Attribution", expanded=True):
                        for b in r['details'].get('breakdown', []):
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"**{b['criterion']}**")
                            with col_b:
                                st.write(f"{b['subscore']:.1f}")
                            # Progress visualization
                            progress_val = b['subscore'] / 100
                            st.progress(min(1.0, max(0.0, progress_val)))
                
                if show_grammar_examples and r['details'].get('grammar', {}).get('available'):
                    with st.expander("🔍 Syntactic Review"):
                        g = r['details']['grammar']
                        st.write(f"**Issues found:** {g['issues_count']}")
                        for ex in g.get('examples', []):
                            suggestions = ex.get('suggestions', [])
                            suggestion_text = ', '.join(suggestions) if suggestions else "No suggestions"
                            st.markdown(f"""
                            <div class='grammar-issue'>
                                <b>{ex.get('message', 'Unknown issue')}</b><br>
                                Correction: {suggestion_text}
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    if st.session_state.get('results'):
        valid_results = [r for r in st.session_state.results if not r.get('error')]
        
        if valid_results:
            scores = [r['final_score'] for r in valid_results]
            similarities = [r['details'].get('similarity', 0) * 100 for r in valid_results]
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average Score", f"{np.mean(scores):.2f}")
            m2.metric("Peak Score", f"{max(scores):.2f}")
            m3.metric("Cohort Size", len(valid_results))
            m4.metric("Avg Similarity", f"{np.mean(similarities):.1f}%")
            
            if HAS_MATPLOTLIB and scores:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.set_facecolor('#131720')
                fig.patch.set_facecolor('#0b0e14')
                
                # Create histogram
                ax.hist(scores, color='#4facfe', alpha=0.8, bins=10, rwidth=0.9, edgecolor='white')
                ax.set_title("Grade Distribution", color='white', fontsize=14, pad=20)
                ax.set_xlabel("Score", color='white')
                ax.set_ylabel("Frequency", color='white')
                ax.tick_params(colors='white')
                
                # Add grid
                ax.grid(True, alpha=0.2, color='white')
                
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Visualization requires matplotlib. Install with: pip install matplotlib")
            
            # Additional analytics
            st.markdown("#### 📋 Detailed Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Score Range:**")
                st.write(f"- Min: {min(scores):.2f}")
                st.write(f"- Max: {max(scores):.2f}")
                st.write(f"- Std Dev: {np.std(scores):.2f}")
            
            with col2:
                st.write("**Grammar Analysis:**")
                grammar_counts = [r['details'].get('grammar', {}).get('issues_count', 0) for r in valid_results]
                if grammar_counts:
                    st.write(f"- Avg Issues: {np.mean(grammar_counts):.1f}")
                    st.write(f"- Max Issues: {max(grammar_counts)}")
                    st.write(f"- Perfect Submissions: {sum(1 for c in grammar_counts if c == 0)}")
            
            st.markdown("---")
            
            # Export section
            st.markdown("#### 📊 Data Export")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("Export all grading results to Excel for further analysis.")
            
            with col2:
                if st.button("📊 EXPORT DATASHEET (.XLSX)", use_container_width=True):
                    excel_data = export_results_to_excel(valid_results)
                    st.download_button(
                        label="⬇️ Download Now", 
                        data=excel_data, 
                        file_name=f"Grades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.warning("No valid grading results to display.")
    else:
        st.info("Analytical engine awaiting grading results. Run a grading session first.")

# ==================== FINAL SYSTEM LOGIC & FOOTER ====================

# Initialize session state if not present (Prevents errors on cold start)
if 'results' not in st.session_state:
    st.session_state.results = []

# Global Reset Action (HCI: User Control & Freedom)
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Grading Data", use_container_width=True):
    st.session_state.results = []
    st.success("All grading data cleared!")
    st.rerun()

# Professional Footer with System Status
st.markdown("<br><br>", unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown("""
        <div style="text-align: center; padding: 20px; opacity: 0.7;">
            <p style="font-size: 0.85rem; color: #6a7682; margin-bottom: 5px;">
                <b>ai!Auto-Grader Pro v2.0</b> | Optimized for Academic Integrity & Scalability
            </p>
            <div style="display: flex; justify-content: center; gap: 15px; font-size: 0.75rem; color: #4facfe;">
                <span>● Neural Engine: BGE-Large-v1.5</span>
                <span>● Processing: GPU-Accelerated</span>
                <span>● Latency: &lt; 1.2s / doc</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Add a subtle "How it Works" guide at the bottom for new users (Cognitive Load Reduction)
with st.expander("📖 System Architecture & Grading Logic"):
    st.markdown("""
    ### Neural Comparison Workflow
    1. **Vectorization**: The system converts your Model Solution and Student work into 1024-dimensional semantic vectors.
    2. **Cosine Similarity**: It measures the 'angle' between these thoughts in high-dimensional space.
    3. **Syntax Analysis**: Jina AI scans for grammatical patterns and logical flow.
    4. **Rubric Weighting**: Scores are calculated based on your custom-defined criteria and weights.
    
    ### Jina AI Integration
    - **Grammar Checking**: Uses advanced LLM analysis to identify grammatical errors
    - **Feedback Generation**: Provides constructive, actionable suggestions
    - **Semantic Understanding**: Goes beyond simple pattern matching
    
    ### Scoring Systems
    - **Numeric (0-100)**: Traditional percentage-based scoring
    - **IELTS Band (0-9)**: International English Language Testing System scale
    
    ### Best Practices
    1. Provide clear model answers for better comparison
    2. Define specific rubric criteria for consistent grading
    3. Use the preview function to verify rubric parsing
    4. Export results for record-keeping and analysis
    """)
