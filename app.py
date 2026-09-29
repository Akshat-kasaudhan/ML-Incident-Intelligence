import streamlit as st
import joblib
import pandas as pd

# Professional Page Config
st.set_page_config(page_title="Enterprise RCA Intelligence", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

# Enterprise SaaS CSS Theme (Clean, Light, Professional)
st.markdown("""
<style>
    /* Clean corporate look */
    .stApp {
        background-color: #f8f9fa;
        color: #202124;
    }
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1a73e8;
        font-weight: 600;
        margin-bottom: 0px;
        padding-top: 10px;
    }
    .sub-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #5f6368;
        font-size: 18px;
        margin-bottom: 40px;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 15px;
        padding-bottom: 15px;
        color: #5f6368;
    }
    .stTabs [aria-selected="true"] {
        color: #1a73e8 !important;
        border-bottom-color: #1a73e8 !important;
    }
    /* Professional Card UI for Results */
    .card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.03);
        border: 1px solid #e0e0e0;
        text-align: center;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
    }
    .card h4 {
        color: #5f6368;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 15px;
        font-weight: 500;
    }
    .card h1 {
        color: #1a73e8;
        font-size: 36px;
        margin: 0;
        font-weight: 700;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        margin-top: 30px;
        margin-bottom: 30px;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #1557b0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏢 Enterprise RCA Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Root Cause Analysis for Microservice Architecture</p>', unsafe_allow_html=True)

try:
    model_service = joblib.load("model_service.pkl")
    model_fault = joblib.load("model_fault.pkl")
    feature_names = joblib.load("feature_names.pkl")
    models_loaded = True
except Exception as e:
    st.error("⚠️ System Offline: ML Models missing from deployment directory.")
    models_loaded = False

if models_loaded:
    # Sidebar Context
    st.sidebar.markdown("### ⚙️ System Status")
    st.sidebar.success("Models Loaded & Active")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Diagnostic Engine Info:**")
    st.sidebar.markdown("This tool runs an inference engine over telemetry deltas to isolate cascading failures.")
    st.sidebar.markdown("*Input the peak metrics recorded post-incident minus the pre-incident baselines.*")
    
    st.markdown("### 📊 Telemetry Input")
    
    input_data = {}
    
    # Beautiful Tabs layout instead of a long messy sidebar
    tab1, tab2, tab3 = st.tabs(["💻 CPU Spikes", "🧠 Memory Spikes", "⚡ Latency Spikes (p90)"])
    
    def render_inputs(tab, keyword):
        with tab:
            st.markdown("<br>", unsafe_allow_html=True)
            # 3 columns for professional compact layout
            cols = st.columns(3)
            col_idx = 0
            for feat in feature_names:
                if keyword in feat:
                    with cols[col_idx % 3]:
                        # Make labels readable: adservice_cpu_spike -> Adservice
                        nice_label = feat.split("_")[0].capitalize()
                        val = st.number_input(f"{nice_label}", value=0.0, step=0.01, key=feat)
                        input_data[feat] = [val]
                    col_idx += 1
                    
    render_inputs(tab1, "cpu")
    render_inputs(tab2, "mem")
    render_inputs(tab3, "lat")
    
    # Safety net
    for feat in feature_names:
        if feat not in input_data:
            input_data[feat] = [0.0]
            
    df_input = pd.DataFrame(input_data)[feature_names]
    
    if st.button("Run Diagnostic Analysis 🚀"):
        with st.spinner("Analyzing telemetry metrics and isolating root cause..."):
            service_pred = model_service.predict(df_input)[0]
            fault_pred = model_fault.predict(df_input)[0]
            
        st.markdown("### 🎯 Diagnostic Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <h4>Suspected Root Cause Service</h4>
                <h1>{service_pred.upper()}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="card">
                <h4>Detected Fault Signature</h4>
                <h1 style="color: #d93025;">{fault_pred.upper()}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        st.success("✅ Diagnostic analysis complete. Acknowledgment logged.")
