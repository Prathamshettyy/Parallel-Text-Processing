"""
COMPLETE STREAMLIT APP - IMPROVED UI WITH CARD-BASED DESIGN
Fixed login caption alignment and modern card-based welcome screen
"""

import streamlit as st
import pandas as pd
import time
import io
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hashlib
import nltk

# Download required NLTK data resources at runtime
nltk_data_packages = ['stopwords', 'punkt', 'wordnet']
for package in nltk_data_packages:
    try:
        nltk.data.find(f'corpus/{package}')
    except LookupError:
        nltk.download(package)


# Import utility functions
try:
    from utils.data_processing import clean_and_normalize_text, load_data_to_db, fetch_data_from_db
    from utils.traditional_model import run_traditional_sentiment_analysis
    from utils.llm_model import run_llm_sentiment_analysis
    from utils.email_sender import send_email_with_attachment
except ImportError as e:
    st.error(f"Error importing utility modules: {e}")
    st.info("Make sure all files in the utils/ folder are created correctly.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'smtp_config' not in st.session_state:
    st.session_state.smtp_config = {'server': 'smtp.gmail.com', 'port': 587}

# Database for users
def init_user_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            smtp_server TEXT DEFAULT 'smtp.gmail.com',
            smtp_port INTEGER DEFAULT 587,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email, password, smtp_server, smtp_port):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        hashed_pw = hash_password(password)
        cursor.execute(
            "INSERT INTO users (name, email, password, smtp_server, smtp_port) VALUES (?, ?, ?, ?, ?)",
            (name, email, hashed_pw, smtp_server, smtp_port)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    hashed_pw = hash_password(password)
    cursor.execute(
        "SELECT name, email, smtp_server, smtp_port FROM users WHERE email = ? AND password = ?",
        (email, hashed_pw)
    )
    result = cursor.fetchone()
    conn.close()
    return result

def logout():
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.user_name = None
    st.session_state.processed_data = None
    st.session_state.results_df = None
    st.session_state.processing_complete = False
    st.session_state.uploaded_df = None

init_user_db()

# ULTRA-MODERN CSS
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #ec4899;
        --accent: #14b8a6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.8);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: rgba(255, 255, 255, 0.1);
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0 0.5rem 0;
        letter-spacing: -2px;
        animation: fadeInDown 0.8s ease;
    }
    
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        position: relative;
        padding-bottom: 0.75rem;
    }
    
    .sub-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 2px;
    }
    
    /* Feature Cards */
    .feature-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        transition: transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.35s, border-color 0.35s;
        height: 100%;
        position: relative;
        will-change: transform, box-shadow;
        transform-origin: center;
        overflow: hidden;
    }
    
    /* subtle animated glow layer */
    .feature-card::before {
        content: '';
        position: absolute;
        inset: -50%;
        background: radial-gradient(600px 200px at 10% 10%, rgba(99,102,241,0.12), transparent 10%),
                    radial-gradient(400px 140px at 90% 90%, rgba(236,72,153,0.09), transparent 10%);
        opacity: 0;
        transform: scale(0.95);
        transition: opacity 0.35s ease, transform 0.35s ease;
        pointer-events: none;
    }
    
    .feature-card:hover::before {
        opacity: 1;
        transform: scale(1);
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02) rotateX(1deg);
        border-color: rgba(99, 102, 241, 0.8);
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.6), 0 6px 18px rgba(99,102,241,0.12);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
        transition: transform 0.35s ease, filter 0.35s ease;
    }
    
    .feature-card:hover .feature-icon {
        transform: translateY(-6px) rotate(-6deg) scale(1.08);
        filter: drop-shadow(0 6px 14px rgba(99,102,241,0.25));
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        transition: color 0.35s ease, transform 0.35s ease;
    }
    
    .feature-card:hover .feature-title {
        color: white;
        transform: translateY(-2px);
    }
    
    .feature-desc {
        font-size: 0.95rem;
        color: var(--text-secondary);
        line-height: 1.6;
        transition: color 0.35s ease;
    }
    
    .feature-card:hover .feature-desc {
        color: #e6eef8;
    }
    
    /* Step Cards */
    .step-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.06) 0%, rgba(236, 72, 153, 0.06) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        position: relative;
        overflow: hidden;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.35s, border-color 0.35s;
        will-change: transform, box-shadow;
        transform-origin: center;
    }
    
    .step-card::before {
        content: '';
        position: absolute;
        inset: -40%;
        background: linear-gradient(90deg, rgba(99,102,241,0.06), rgba(20,184,166,0.03), rgba(236,72,153,0.05));
        opacity: 0;
        transform: translateY(8px);
        transition: opacity 0.35s ease, transform 0.35s ease;
        pointer-events: none;
    }
    
    .step-card:hover::before {
        opacity: 1;
        transform: translateY(0);
    }
    
    .step-card:hover {
        transform: translateY(-8px) scale(1.01) rotateX(1.2deg);
        border-color: rgba(236, 72, 153, 0.6);
        box-shadow: 0 18px 44px rgba(15,23,42,0.55), 0 6px 20px rgba(236,72,153,0.08);
    }
    
    .step-number {
        position: absolute;
        top: -10px;
        right: 10px;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        opacity: 0.2;
        transition: opacity 0.35s, transform 0.35s;
    }
    
    .step-card:hover .step-number {
        opacity: 0.35;
        transform: translateY(-6px) rotate(-2deg);
    }
    
    .step-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        transition: color 0.35s;
    }
    
    .step-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        transition: color 0.35s;
    }
    
    .step-card:hover .step-title { color: #ffffff; }
    .step-card:hover .step-desc { color: #dbeafe; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border-bottom: 2px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        color: var(--text-secondary);
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%);
        color: var(--text-primary);
        border-bottom: 3px solid var(--primary);
    }
    
    [data-testid="stMetric"] {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem;
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text-primary);
        padding: 0.875rem 1rem;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
        border-radius: 10px;
    }
    
    [data-testid="stFileUploader"] {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 2px dashed var(--border);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background: rgba(99, 102, 241, 0.05);
    }
    
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--success);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--danger);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--primary), var(--secondary), transparent);
        margin: 2rem 0;
        opacity: 0.5;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stCaption {
        color: var(--text-secondary);
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .streamlit-expanderHeader {
        background: var(--glass-bg);
        border-radius: 12px;
        border: 1px solid var(--border);
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: var(--primary);
    }
    
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.05); border-radius: 10px; }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 10px;
    }
</style>
''', unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None

# Helper functions
def load_csv_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if df.empty:
            raise ValueError("DataFrame is empty")
        return df
    except:
        try:
            uploaded_file.seek(0)
            bytes_data = uploaded_file.getvalue()
            df = pd.read_csv(io.BytesIO(bytes_data))
            return df
        except Exception as e:
            raise Exception(f"Could not read CSV file: {str(e)}")

def plot_confusion_matrix(cm, title, labels=None):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=2, linecolor='#475569',
                cbar_kws={'label': 'Count'}, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold', 'color': '#f8fafc'})
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#f8fafc')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', color='#cbd5e1')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', color='#cbd5e1')
    
    if labels:
        ax.set_xticklabels(labels, rotation=45, ha='right', color='#cbd5e1')
        ax.set_yticklabels(labels, rotation=0, color='#cbd5e1')
    
    ax.tick_params(colors='#cbd5e1')
    plt.tight_layout()
    return fig

def plot_comparison_chart(trad_time, llm_time, trad_acc, llm_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f172a')
    ax1.set_facecolor('#1e293b')
    ax2.set_facecolor('#1e293b')
    
    methods = ['Traditional', 'LLM']
    times = [trad_time, llm_time]
    colors = ['#6366f1', '#ec4899']
    
    bars1 = ax1.bar(methods, times, color=colors, alpha=0.8, edgecolor='#475569', linewidth=2)
    ax1.set_ylabel('Time (seconds)', fontweight='bold', color='#cbd5e1')
    ax1.set_title('Speed Comparison', fontweight='bold', fontsize=14, color='#f8fafc', pad=15)
    ax1.grid(axis='y', alpha=0.2, color='#475569')
    ax1.tick_params(colors='#cbd5e1')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}s',
                ha='center', va='bottom', fontweight='bold', color='#f8fafc')
    
    accuracies = [trad_acc * 100, llm_acc * 100]
    bars2 = ax2.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='#475569', linewidth=2)
    ax2.set_ylabel('Accuracy (%)', fontweight='bold', color='#cbd5e1')
    ax2.set_title('Accuracy Comparison', fontweight='bold', fontsize=14, color='#f8fafc', pad=15)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.2, color='#475569')
    ax2.tick_params(colors='#cbd5e1')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', color='#f8fafc')
    
    plt.tight_layout()
    return fig

def generate_summary_report(traditional_results, llm_results, processing_method):
    report = f'''
SENTIMENT ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
User: {st.session_state.user_name} ({st.session_state.user_email})
{'=' * 80}

PROCESSING METHOD: {processing_method}
{'=' * 80}

'''
    if traditional_results:
        report += f'''
TRADITIONAL METHOD
{'-' * 80}
Time: {traditional_results['time']:.2f}s
Accuracy: {traditional_results.get('accuracy', 0):.2%}
Samples: {len(traditional_results['predictions'])}

{traditional_results.get('classification_report', 'N/A')}

'''
    
    if llm_results:
        report += f'''
LLM METHOD
{'-' * 80}
Time: {llm_results['time']:.2f}s
Accuracy: {llm_results.get('accuracy', 0):.2%}
Samples: {len(llm_results['predictions'])}

{llm_results.get('classification_report', 'N/A')}

'''
    
    if traditional_results and llm_results:
        speedup = traditional_results['time'] / llm_results['time'] if llm_results['time'] > 0 else 0
        acc_diff = llm_results.get('accuracy', 0) - traditional_results.get('accuracy', 0)
        report += f'''
COMPARISON
{'-' * 80}
Speed: {'LLM' if speedup > 1 else 'Traditional'} is {abs(speedup):.2f}x faster
Accuracy: LLM is {acc_diff:.2%} {'more' if acc_diff > 0 else 'less'} accurate

'''
    
    return report

# Login Page
def show_login_page():
    st.markdown('<h1 class="main-header">Sentiment Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="stCaption">Advanced AI-Powered Text Analysis</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
        
        with tab1:
            st.subheader("Welcome Back")
            login_email = st.text_input("Email", key="login_email", placeholder="your.email@example.com")
            login_password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")
            
            if st.button("Sign In", key="login_btn", use_container_width=True):
                if login_email and login_password:
                    user_data = login_user(login_email, login_password)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user_name = user_data[0]
                        st.session_state.user_email = user_data[1]
                        st.session_state.smtp_config = {'server': user_data[2], 'port': user_data[3]}
                        st.success(f"Welcome, {user_data[0]}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                else:
                    st.warning("Enter email and password")
        
        with tab2:
            st.subheader("Create Account")
            reg_name = st.text_input("Full Name", key="reg_name", placeholder="John Doe")
            reg_email = st.text_input("Email", key="reg_email", placeholder="john@example.com")
            reg_password = st.text_input("Password", type="password", key="reg_password", placeholder="••••••••")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm", placeholder="••••••••")
            
            with st.expander("Email Settings (Optional)"):
                smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", value=587)
            
            if st.button("Sign Up", key="register_btn", use_container_width=True):
                if reg_name and reg_email and reg_password and reg_confirm:
                    if reg_password == reg_confirm:
                        if register_user(reg_name, reg_email, reg_password, smtp_server, smtp_port):
                            st.success("Account created! Please sign in.")
                        else:
                            st.error("Email already exists")
                    else:
                        st.error("Passwords don't match")
                else:
                    st.warning("Fill all fields")

# Main Application - Continue in next part...
# PART 2: Main Application with Card-Based Welcome Screen

def main():
    if st.session_state.logged_in:
        st.markdown('<h1 class="main-header">Sentiment Analysis Platform</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="stCaption">Welcome back, {st.session_state.user_name}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<h1 class="main-header">Sentiment Analysis Platform</h1>', unsafe_allow_html=True)
        st.markdown('<p class="stCaption">Advanced AI-Powered Sentiment Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Configuration")
        
        if st.session_state.logged_in:
            if st.button("Sign Out", key="logout_btn"):
                logout()
                st.rerun()
            st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            st.success(f"✓ {uploaded_file.name}")
            file_size = len(uploaded_file.getvalue()) / 1024
            st.caption(f"{file_size:.2f} KB")
            
            try:
                if st.session_state.uploaded_df is None or st.button("Reload"):
                    with st.spinner("Loading..."):
                        st.session_state.uploaded_df = load_csv_file(uploaded_file)
                        st.success(f"✓ {len(st.session_state.uploaded_df):,} rows")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("Processing Options")
        text_column = st.text_input("Text Column", value="review")
        label_column = st.text_input("Label Column", value="sentiment")
        sample_size = st.number_input("Sample Size", min_value=100, max_value=50000, value=1000, step=100)
        processing_method = st.radio("Method", options=["Both (Compare)", "Traditional Only", "LLM Only"])
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("Email Results")
        send_email = st.checkbox("Send via email")
        
        email_address = None
        smtp_server = st.session_state.smtp_config['server'] if st.session_state.logged_in else "smtp.gmail.com"
        smtp_port = st.session_state.smtp_config['port'] if st.session_state.logged_in else 587
        sender_email = None
        sender_password = None
        
        if send_email:
            if st.session_state.logged_in:
                email_address = st.session_state.user_email
                st.info(f"To: {email_address}")
                sender_email = st.text_input("SMTP Email", value=st.session_state.user_email)
                sender_password = st.text_input("Password", type="password")
            else:
                email_address = st.text_input("Email")
                sender_email = st.text_input("SMTP Email")
                sender_password = st.text_input("Password", type="password")
    
    if uploaded_file is not None and st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        
        tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Processing", "Results", "Download"])
        
        with tab1:
            st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Sample", f"{min(sample_size, len(df)):,}")
            
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True, height=400)
            
            st.subheader("Column Details")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True)
            
            if text_column in df.columns:
                st.subheader("Text Statistics")
                df_copy = df.copy()
                df_copy['text_length'] = df_copy[text_column].astype(str).str.len()
                df_copy['word_count'] = df_copy[text_column].astype(str).str.split().str.len()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Length", f"{df_copy['text_length'].mean():.0f}")
                with col2:
                    st.metric("Avg Words", f"{df_copy['word_count'].mean():.0f}")
                with col3:
                    st.metric("Min Words", f"{df_copy['word_count'].min():.0f}")
                with col4:
                    st.metric("Max Words", f"{df_copy['word_count'].max():.0f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f172a')
                ax1.set_facecolor('#1e293b')
                ax2.set_facecolor('#1e293b')
                
                df_copy['text_length'].hist(bins=50, ax=ax1, color='#6366f1', alpha=0.7, edgecolor='#475569')
                ax1.set_title('Text Length', fontweight='bold', color='#f8fafc')
                ax1.set_xlabel('Characters', color='#cbd5e1')
                ax1.set_ylabel('Frequency', color='#cbd5e1')
                ax1.grid(alpha=0.2, color='#475569')
                ax1.tick_params(colors='#cbd5e1')
                
                df_copy['word_count'].hist(bins=50, ax=ax2, color='#ec4899', alpha=0.7, edgecolor='#475569')
                ax2.set_title('Word Count', fontweight='bold', color='#f8fafc')
                ax2.set_xlabel('Words', color='#cbd5e1')
                ax2.set_ylabel('Frequency', color='#cbd5e1')
                ax2.grid(alpha=0.2, color='#475569')
                ax2.tick_params(colors='#cbd5e1')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab2:
            st.markdown('<h2 class="sub-header">Process Data</h2>', unsafe_allow_html=True)
            
            if st.button("Start Processing", type="primary", use_container_width=True):
                try:
                    df_process = df.copy()
                    
                    if len(df_process) > sample_size:
                        df_process = df_process.sample(n=sample_size, random_state=42)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Step 1/5: Cleaning text...")
                    progress_bar.progress(10)
                    
                    if text_column not in df_process.columns:
                        st.error(f"Column '{text_column}' not found!")
                        st.stop()
                    
                    df_process['cleaned_text'] = df_process[text_column].apply(clean_and_normalize_text)
                    progress_bar.progress(20)
                    st.success("Text cleaned")
                    
                    status_text.text("Step 2/5: Saving database...")
                    progress_bar.progress(30)
                    load_data_to_db(df_process, 'sentiment_analysis.db')
                    progress_bar.progress(40)
                    st.success("Database saved")
                    
                    traditional_results = None
                    if processing_method in ["Both (Compare)", "Traditional Only"]:
                        status_text.text("Step 3/5: Traditional analysis...")
                        progress_bar.progress(50)
                        
                        traditional_results = run_traditional_sentiment_analysis(
                            df_process['cleaned_text'].tolist(),
                            df_process[label_column].tolist() if label_column in df_process.columns else None
                        )
                        
                        progress_bar.progress(65)
                        st.success(f"Traditional: {traditional_results['time']:.2f}s")
                    
                    llm_results = None
                    if processing_method in ["Both (Compare)", "LLM Only"]:
                        status_text.text("Step 4/5: LLM analysis...")
                        progress_bar.progress(70)
                        
                        llm_results = run_llm_sentiment_analysis(
                            df_process['cleaned_text'].tolist(),
                            df_process[label_column].tolist() if label_column in df_process.columns else None
                        )
                        
                        progress_bar.progress(90)
                        st.success(f"LLM: {llm_results['time']:.2f}s")
                    
                    status_text.text("Step 5/5: Compiling...")
                    progress_bar.progress(95)
                    
                    results_df = df_process.copy()
                    
                    if processing_method in ["Both (Compare)", "Traditional Only"]:
                        results_df['traditional_sentiment'] = traditional_results['predictions']
                    
                    if processing_method in ["Both (Compare)", "LLM Only"]:
                        results_df['llm_sentiment'] = llm_results['predictions']
                    
                    st.session_state.results_df = results_df
                    st.session_state.traditional_results = traditional_results
                    st.session_state.llm_results = llm_results
                    st.session_state.processing_complete = True
                    st.session_state.processing_method = processing_method
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.balloons()
                    st.success("Processing complete!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        with tab3:
            st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
            
            if st.session_state.processing_complete:
                results_df = st.session_state.results_df
                processing_method = st.session_state.processing_method
                
                if processing_method == "Both (Compare)":
                    st.subheader("Performance Metrics")
                    
                    trad_time = st.session_state.traditional_results['time']
                    llm_time = st.session_state.llm_results['time']
                    trad_acc = st.session_state.traditional_results.get('accuracy', 0)
                    llm_acc = st.session_state.llm_results.get('accuracy', 0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Traditional", f"{trad_time:.2f}s", "Speed")
                    with col2:
                        st.metric("LLM", f"{llm_time:.2f}s", "Speed")
                    with col3:
                        st.metric("Traditional", f"{trad_acc:.2%}", "Accuracy")
                    with col4:
                        st.metric("LLM", f"{llm_acc:.2%}", "Accuracy")
                    
                    if label_column in results_df.columns:
                        fig = plot_comparison_chart(trad_time, llm_time, trad_acc, llm_acc)
                        st.pyplot(fig)
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.subheader("Confusion Matrices")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'confusion_matrix' in st.session_state.traditional_results:
                                cm = st.session_state.traditional_results['confusion_matrix']
                                fig = plot_confusion_matrix(cm, "Traditional Method")
                                st.pyplot(fig)
                        
                        with col2:
                            if 'confusion_matrix' in st.session_state.llm_results:
                                cm = st.session_state.llm_results['confusion_matrix']
                                fig = plot_confusion_matrix(cm, "LLM Method")
                                st.pyplot(fig)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("Predictions Sample")
                
                display_cols = [text_column]
                if label_column in results_df.columns:
                    display_cols.append(label_column)
                if processing_method in ["Both (Compare)", "Traditional Only"]:
                    display_cols.append('traditional_sentiment')
                if processing_method in ["Both (Compare)", "LLM Only"]:
                    display_cols.append('llm_sentiment')
                
                st.dataframe(results_df[display_cols].head(20), use_container_width=True, height=400)
                
                with st.expander("Classification Reports"):
                    if processing_method in ["Both (Compare)", "Traditional Only"]:
                        st.subheader("Traditional")
                        st.code(st.session_state.traditional_results.get('classification_report', 'N/A'))
                    
                    if processing_method in ["Both (Compare)", "LLM Only"]:
                        st.subheader("LLM")
                        st.code(st.session_state.llm_results.get('classification_report', 'N/A'))
            else:
                st.info("Process data first in Processing tab")
        
        with tab4:
            st.markdown('<h2 class="sub-header">Export Results</h2>', unsafe_allow_html=True)
            
            if st.session_state.processing_complete:
                results_df = st.session_state.results_df
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                with col2:
                    report = generate_summary_report(
                        st.session_state.traditional_results if st.session_state.processing_method in ["Both (Compare)", "Traditional Only"] else None,
                        st.session_state.llm_results if st.session_state.processing_method in ["Both (Compare)", "LLM Only"] else None,
                        st.session_state.processing_method
                    )
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                        mime='text/plain',
                        use_container_width=True
                    )
                
                if send_email and email_address:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    if st.button("Send Email", use_container_width=True):
                        try:
                            with st.spinner("Sending..."):
                                send_email_with_attachment(
                                    sender_email=sender_email,
                                    sender_password=sender_password,
                                    recipient_email=email_address,
                                    subject="Sentiment Analysis Results",
                                    body=report,
                                    attachment_data=csv,
                                    attachment_name=f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                                    smtp_server=smtp_server,
                                    smtp_port=smtp_port
                                )
                            st.success(f"Sent to {email_address}")
                        except Exception as e:
                            st.error(f"Failed: {str(e)}")
            else:
                st.info("Process data first")
    
    else:
        # IMPROVED WELCOME SCREEN WITH CARDS
        st.markdown('<h2 class="sub-header">Platform Features</h2>', unsafe_allow_html=True)
        
        # Feature Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('''
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <div class="feature-title">Data Processing</div>
                <div class="feature-desc">Upload CSV files with text reviews and automatically clean, normalize, and prepare data for analysis</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="feature-card">
                <span class="feature-icon">🤖</span>
                <div class="feature-title">Dual Analysis</div>
                <div class="feature-desc">Compare traditional TextBlob sentiment analysis with advanced DistilBERT deep learning models</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div class="feature-card">
                <span class="feature-icon">⚡</span>
                <div class="feature-title">GPU Acceleration</div>
                <div class="feature-desc">Leverage parallel processing for traditional methods and GPU acceleration for LLM inference</div>
            </div>
            ''', unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown('''
            <div class="feature-card">
                <span class="feature-icon">📈</span>
                <div class="feature-title">Performance Metrics</div>
                <div class="feature-desc">Compare speed, accuracy, and detailed classification reports with confusion matrices</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col5:
            st.markdown('''
            <div class="feature-card">
                <span class="feature-icon">💾</span>
                <div class="feature-title">Export Results</div>
                <div class="feature-desc">Download results as CSV files and comprehensive text reports with all analysis details</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col6:
            st.markdown('''
            <div class="feature-card">
                <span class="feature-icon">📧</span>
                <div class="feature-title">Email Integration</div>
                <div class="feature-desc">Send analysis results and reports directly to your email with customizable SMTP settings</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Getting Started</h2>', unsafe_allow_html=True)
        
        # Step Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('''
            <div class="step-card">
                <div class="step-number">1</div>
                <div class="step-title">Upload Dataset</div>
                <div class="step-desc">Upload your CSV file with text data using the sidebar file uploader</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="step-card">
                <div class="step-number">3</div>
                <div class="step-title">Select Analysis Method</div>
                <div class="step-desc">Choose between Traditional, LLM, or Both methods for comparison</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="step-card">
                <div class="step-number">5</div>
                <div class="step-title">Download & Share</div>
                <div class="step-desc">Export results as CSV or reports, and optionally send via email</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="step-card">
                <div class="step-number">2</div>
                <div class="step-title">Configure Options</div>
                <div class="step-desc">Set text column name, label column, and sample size for processing</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="step-card">
                <div class="step-number">4</div>
                <div class="step-title">Process & Analyze</div>
                <div class="step-desc">Click "Start Processing" and view real-time progress through 5 steps</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="step-card">
                <div class="step-number">6</div>
                <div class="step-title">View Results</div>
                <div class="step-desc">Explore performance metrics, confusion matrices, and sample predictions</div>
            </div>
            ''', unsafe_allow_html=True)

if __name__ == "__main__":
    if st.session_state.logged_in:
        main()
    else:
        show_login_page()

