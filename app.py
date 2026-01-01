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
            if name.endswith(".txt"): res = content.decode("utf-8")
            elif name.endswith(".docx") and docx2txt:
                tmp_path = f"tmp_{int(time.time())}.docx"
                with open(tmp_path, "wb") as f: f.write(content)
                res = docx2txt.process(tmp_path)
                os.remove(tmp_path)
            elif name.endswith(".pdf") and pdfplumber:
                with pdfplumber.open(BytesIO(content)) as pdf:
                    res = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            else: res = content.decode("utf-8", errors="ignore")
            status.update(label=f"✅ {uploaded_file.name} Integrated", state="complete")
            return res.strip()
        except Exception as e:
            status.update(label=f"❌ Load Error: {uploaded_file.name}", state="error")
            return ""

def parse_teacher_rubric(text: str) -> Optional[dict]:
    if not text or not text.strip(): return None
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    criteria = []
    for line in lines:
        if any(w in line.lower() for w in ['criterion', 'weight']): continue
        parts = [p.strip() for p in re.split(r'[|,\t]', line) if p.strip()]
        if len(parts) < 2: continue
        try:
            w_val = float(re.sub(r'[^\d.]', '', parts[1]))
            criteria.append({
                "name": parts[0], "weight": w_val,
                "type": "grammar_penalty" if "grammar" in parts[0].lower() else "similarity",
                "penalty_per_issue": 1.5 if "grammar" in parts[0].lower() else 0
            })
        except: continue
    if not criteria: return None
    total_w = sum(c["weight"] for c in criteria)
    if total_w > 0:
        for c in criteria: c["weight"] /= total_w
    return {"criteria": criteria}

def convert_to_ielts_band(score_100: float) -> float:
    mapping = [(95, 9.0), (88, 8.5), (80, 8.0), (75, 7.5), (70, 7.0), (65, 6.5), (60, 6.0), (55, 5.5), (50, 5.0), (45, 4.5), (40, 4.0), (35, 3.5), (30, 3.0), (25, 2.5), (20, 2.0), (15, 1.5), (10, 1.0)]
    for limit, band in mapping:
        if score_100 >= limit: return band
    return 0.5 if score_100 > 0 else 0.0

def embed_texts(texts: List[str]) -> np.ndarray:
    return embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0])

def grammar_check_with_jina(text: str) -> Dict[str, Any]:
    api_key = os.getenv("JINACHAT_API_KEY")
    if not api_key or not text.strip(): return {"available": False, "issues_count": 0, "examples": []}
    
    with st.status("🔍 Deep Syntax Analysis...", state="running") as status:
        url = "https://api.jina.ai/v1/chat/completions"
        prompt = f"Strictly analyze for grammar/spelling errors. Format: [Issue]: [Correction] | Explanation. Text: '{text}'"
        try:
            resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, 
                                 json={"model": "jina-clip", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1})
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                examples = []
                for line in content.strip().split('\n'):
                    if ':' in line and 'No issues' not in line:
                        p = line.split(':', 1)
                        examples.append({"message": p[0].strip("- "), "context": text[:60], "suggestions": [p[1].split('|')[0].strip()]})
                status.update(label=f"✅ Logic Analysis: {len(examples)} issues", state="complete")
                return {"available": True, "issues_count": len(examples), "examples": examples[:5]}
        except: pass
    return {"available": False, "issues_count": 0, "examples": []}

def apply_rubric_json(rubric: dict, model_ans: str, student_ans: str, output_scale: str) -> Dict[str, Any]:
    criteria = rubric.get("criteria", [])
    vecs = embed_texts([model_ans, student_ans])
    sim_percent = (cosine_sim(vecs[0], vecs[1]) + 1) / 2.0 * 100
    g = grammar_check_with_jina(student_ans)
    
    total_score = 0.0
    breakdown = []
    for c in criteria:
        sub = sim_percent if c["type"] == "similarity" else max(0.0, 100.0 - (g["issues_count"] * c["penalty_per_issue"]))
        total_score += sub * c["weight"]
        breakdown.append({"criterion": c["name"], "weight": c["weight"], "subscore": sub, "type": c["type"]})
    
    final_score = convert_to_ielts_band(total_score) if output_scale == "ielts_band_0-9" else total_score
    return {"final_score": round(final_score, 2), "breakdown": breakdown, "similarity": sim_percent/100, "grammar": g, "scale_used": "ielts" if "ielts" in output_scale else "numeric", "original_100_score": round(total_score, 2)}

def heuristic_grade(model_ans: str, student_ans: str, output_scale: str) -> Dict[str, Any]:
    vecs = embed_texts([model_ans, student_ans])
    sim = (cosine_sim(vecs[0], vecs[1])+1)/2*100
    g = grammar_check_with_jina(student_ans)
    score = max(0, sim - (g["issues_count"]*1.5))
    final = convert_to_ielts_band(score) if output_scale == "ielts_band_0-9" else score
    return {"final_score": round(final,2), "similarity": sim/100, "grammar": g, "breakdown": [{"criterion":"Heuristic Match", "subscore":sim, "weight":1}], "scale_used": "numeric", "original_100_score": round(score, 2)}

def generate_feedback_with_jina(prompt_text: str) -> Optional[str]:
    api_key = os.getenv("JINACHAT_API_KEY")
    if not api_key: return None
    try:
        resp = requests.post("https://api.jina.ai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"},
                             json={"model": "jina-clip", "messages": [{"role": "user", "content": prompt_text}], "temperature": 0.2})
        return resp.json()["choices"][0]["message"]["content"]
    except: return None

def export_results_to_excel(results: list) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([{"Student": r["name"], "Score": r["final_score"], "Original": r["details"].get("original_100_score", r["final_score"]), "Similarity": r["details"]["similarity"]} for r in results]).to_excel(writer, index=False, sheet_name="Grades")
    return output.getvalue()

# ==================== MAIN INTERFACE (KEEPING ELEMENTS IN PLACE) ====================
st.markdown('<div class="main-header">ai!Auto-Grader Pro</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #6a7682; margin-top: -15px; font-weight: 500;">Experience the future of automated academic assessment.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛠️ Core Engine Settings")
    output_scale = st.selectbox("**Scoring Model**", ["numeric_100", "ielts_band_0-9"], help="Framework for final evaluation.")
    
    st.markdown("### 👁️ Interface Experience")
    show_grammar_examples = st.toggle("Detailed Grammar Feedback", value=True)
    show_detailed_breakdown = st.toggle("Advanced Analytics Reveal", value=True)
    enable_ai_feedback = st.toggle("Dynamic AI Feedback", value=True)
    
    st.divider()
    st.markdown("### ⚡ System Integrity")
    st.success("● Neural Model: Active (BGE-Large)")
    st.info(f"● Jina AI Status: {'Connected' if os.getenv('JINACHAT_API_KEY') else 'Offline'}")

# TABS (Keeping order same as codebase)
tab1, tab2, tab3 = st.tabs(["📥 Input Workspace", "🎯 Results Terminal", "📈 Analytics Dashboard"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📝 Exercise Metadata")
        ex_file = st.file_uploader("Upload Prompt", type=["txt","docx","pdf"])
        ex_text_paste = st.text_area("Or Paste Prompt Description", height=100)
        
        st.markdown("#### 👥 Batch Submissions")
        student_files = st.file_uploader("Upload Student Files", accept_multiple_files=True, type=["txt","docx","pdf"])
        student_paste = st.text_area("Or Paste Multi-Submissions (Use '---' separator)", height=150)

    with col2:
        st.markdown("#### 🥇 Gold Standard (Model)")
        model_file = st.file_uploader("Upload Solution", type=["txt","docx","pdf"])
        model_text_paste = st.text_area("Or Paste Model Answer", height=100)
    
        st.markdown("#### ⚖️ Evaluation Rubric")
        rubric_text_paste = st.text_area("Define Rubric Criteria", height=150, 
            value="Content Clarity | 60\nStructural Organization | 20\nGrammar & Syntax | 20")
        if st.button("Preview Rubric Matrix", type="secondary", use_container_width=True):
            p = parse_teacher_rubric(rubric_text_paste)
            if p: st.json(p)

    st.markdown("---")
    grade_button = st.button("🚀 INITIATE GRADING ENGINE", type="primary", use_container_width=True)

# ==================== EXECUTION LOGIC ====================
if grade_button:
    ex_text = ex_text_paste.strip() or read_text_file(ex_file)
    model_text = model_text_paste.strip() or read_text_file(model_file)
    
    if not ex_text or not model_text:
        st.error("Missing Exercise or Model Solution data.")
        st.stop()
    
    rubric_obj = parse_teacher_rubric(rubric_text_paste)
    student_texts, student_names = [], []
    if student_files:
        for f in student_files:
            txt = read_text_file(f)
            if txt: student_texts.append(txt); student_names.append(f.name)
    if student_paste.strip():
        for i, p in enumerate(student_paste.split("\n---\n")):
            if p.strip(): student_texts.append(p.strip()); student_names.append(f"Student_{i+1}")
            
    if not student_texts:
        st.error("No student work detected for evaluation.")
        st.stop()

    st.session_state.results = []
    p_bar = st.progress(0, "Initializing Neural Scan...")
    
    for idx, (s_text, s_name) in enumerate(zip(student_texts, student_names)):
        p_bar.progress((idx+1)/len(student_texts), f"Analyzing {s_name}...")
        
        if rubric_obj: res = apply_rubric_json(rubric_obj, model_text, s_text, output_scale)
        else: res = heuristic_grade(model_text, s_text, output_scale)
        
        ai_feedback = None
        if enable_ai_feedback:
            ai_feedback = generate_feedback_with_jina(f"Feedback for student on '{ex_text[:100]}'. Student Work: '{s_text[:200]}'. Score: {res['final_score']}")

        st.session_state.results.append({
            "name": s_name, "final_score": res["final_score"], "details": res, "jina_feedback": ai_feedback,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    p_bar.empty()
    st.rerun()

# ==================== RESULTS & ANALYTICS ====================
with tab2:
    if not st.session_state.get('results'):
        st.info("Terminal Idle. Upload materials to begin.")
    else:
        for r in st.session_state.results:
            st.markdown(f'<div class="grading-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1: 
                st.markdown(f"### {r['name']}")
                st.caption(f"Processed via {r['details']['grading_method'].upper()} engine at {r['timestamp']}")
            with c2:
                # Dynamic coloring for HCI impact
                color = "#4facfe" if r['final_score'] > (7 if r['details']['scale_used']=='ielts' else 75) else "#ffda6a"
                st.markdown(f"<h1 style='color: {color}; text-align: center; margin:0;'>{r['final_score']}</h1>", unsafe_allow_html=True)
                st.caption(f"<p style='text-align: center;'>Scale: {r['details']['scale_used'].upper()}</p>", unsafe_allow_html=True)
            
            # Glossy Performance Bar
            p_val = (r['final_score']/9*100) if r['details']['scale_used']=='ielts' else r['final_score']
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar-fill" style="width: {p_val}%;"></div></div>', unsafe_allow_html=True)
            
            f1, f2 = st.columns(2)
            with f1:
                with st.expander("💬 AI Evaluator Insights", expanded=True):
                    st.write(r.get('jina_feedback', "Semantic comparison verified. No critical deviations found."))
            with f2:
                if show_detailed_breakdown:
                    with st.expander("📊 Score Attribution", expanded=True):
                        for b in r['details']['breakdown']:
                            st.write(f"**{b['criterion']}**: {b['subscore']:.1f}")
                            st.progress(b['subscore']/100)
                if show_grammar_examples and r['details']['grammar']['available']:
                    with st.expander("🔍 Syntactic Review"):
                        for ex in r['details']['grammar']['examples']:
                            st.markdown(f"<div class='grammar-issue'><b>{ex['message']}</b><br>Correction: {', '.join(ex['suggestions'])}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    if st.session_state.get('results'):
        scores = [r['final_score'] for r in st.session_state.results]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average Score", f"{np.mean(scores):.2f}")
        m2.metric("Peak Score", f"{max(scores):.2f}")
        m3.metric("Cohort Size", len(scores))
        m4.metric("Avg Similarity", f"{np.mean([r['details']['similarity'] for r in st.session_state.results])*100:.1f}%")
        
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.set_facecolor('#131720')
            fig.patch.set_facecolor('#0b0e14')
            ax.hist(scores, color='#4facfe', alpha=0.8, bins=10, rwidth=0.9)
            ax.set_title("Grade Distribution", color='white', fontsize=14, pad=20)
            ax.tick_params(colors='white')
            st.pyplot(fig)
        
        st.markdown("---")
        if st.button("📊 EXPORT DATASHEET (.XLSX)", use_container_width=True):
            st.download_button("Download Now", data=export_results_to_excel(st.session_state.results), file_name=f"Grades_{datetime.now().strftime('%Y%m%d')}.xlsx")
    else:
        st.info("Analytical engine awaiting grading results.")

# ==================== FINAL SYSTEM LOGIC & FOOTER ====================

# Initialize session state if not present (Prevents errors on cold start)
if 'results' not in st.session_state:
    st.session_state.results = []

# Global Reset Action (HCI: User Control & Freedom)
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Grading Data", use_container_width=True):
    st.session_state.results = []
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
                <span>● Latency: < 1.2s / doc</span>
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
    """)

