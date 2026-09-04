import streamlit as st
import pandas as pd

from src.features import engineer_features
from src.detector import score_transaction
from src.investigation_agent import build_investigation_agent
from src.prioritization_agent import assign_ranks
from src.graph import build_graph
from textwrap import dedent 

# Page Configuration
st.set_page_config(page_title="Agentic AML Triage", layout="wide", initial_sidebar_state="expanded")

# Load External CSS Stylesheet
def load_css(file_name="styles.css"):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("globals.css")
load_css("styles.css")

# Header Bar 
st.markdown("""
<div class="dash-header">
    <div class="dash-title">AML Triage Workstation</div>
    <div class="dash-subtitle">
        Automated fraud detection, agentic investigation, and case prioritization
    </div>
</div>
<div class="header-divider"></div>
""", unsafe_allow_html=True)

# Sidebar: Data Loading and Configuration
st.sidebar.markdown("### Control Panel")
st.sidebar.markdown("Manage dataset sources and execution batch sizes.")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
batch_size = st.sidebar.slider("Sample Batch Size", min_value=3, max_value=20, value=5)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
run_btn = st.sidebar.button("Run AML Triage Pipeline")

def prepare_data(file_source):
    """Load, engineer features, and score incoming transactions."""
    if file_source is not None:
        df = pd.read_csv(file_source)
    else:
        df = pd.read_csv("data/paysim_sample.csv")
    
    df = engineer_features(df)
    df["score"] = df.apply(score_transaction, axis=1)
    return df

# Batch execution logic 
if run_btn:
    with st.spinner("Processing dataset and initializing agents..."):
        df = prepare_data(uploaded_file)
        flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])
        
        agent = build_investigation_agent(df, flagged_ids)
        graph = build_graph(df, flagged_ids, agent)
        
        candidates = df[df["score"] >= 0.3]
        if len(candidates) == 0:
            st.warning("No transactions exceeded the risk threshold in the uploaded dataset.")
            st.stop()
            
        sample_batch = candidates.sample(n=min(batch_size, len(candidates)), random_state=1)
        
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (_, row) in enumerate(sample_batch.iterrows()):
        status_text.text(f"Investigating Account {row['nameDest']} ({idx+1}/{len(sample_batch)})...")
        
        state = {
            "transaction": row.to_dict(),
            "recipient_id": row["nameDest"],
            "rewrite_count": 0,
        }
        
        final_state = graph.invoke(state)
        if final_state.get("urgency_score") is not None:
            results.append(final_state)
            
        progress_bar.progress((idx + 1) / len(sample_batch))

    st.toast("Pipeline execution complete", icon=":material/check_circle:")

    progress_bar.empty()
    status_text.empty()
    
    # Store processed results in Session State
    st.session_state["executed"] = True
    st.session_state["raw_results"] = results
    
    # Calculate Ranks across the batch
    rank_tuples = [(fs["investigation_verdict"].account_id, fs["urgency_score"]) for fs in results]
    ranked_list = assign_ranks(rank_tuples)
    st.session_state["ranked"] = ranked_list

# Dashboard Visualization 
if st.session_state.get("executed"):
    results = st.session_state["raw_results"]
    ranked = st.session_state["ranked"]

    # KPI Row 
    total_batch = len(results)
    auto_cleared = sum(1 for r in results if r["detector_verdict"].route == "auto_close")
    escalated = sum(1 for r in results if r["investigation_verdict"].route == "escalate")
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Batch Evaluated</div>
            <div class="kpi-value">{total_batch}</div>
            <div class="kpi-subtext">Transactions Ingested</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Auto-Cleared (Detector)</div>
            <div class="kpi-value">{auto_cleared}</div>
            <div class="kpi-subtext kpi-subtext--success">Reduced Analyst Volume</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Escalated to Queue</div>
            <div class="kpi-value">{escalated}</div>
            <div class="kpi-subtext kpi-subtext--danger">Requires Review</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prioritized Queue Panel
    st.markdown('<div class="dash-card-title">Prioritized Case Queue</div>', unsafe_allow_html=True)
    
    queue_rows = []
    for rank in ranked:
        matching_state = next(fs for fs in results if fs["investigation_verdict"].account_id == rank.account_id)
        inv_v = matching_state["investigation_verdict"]
        
        queue_rows.append({
            "Rank": rank.rank,
            "Account ID": rank.account_id,
            "Urgency Score": rank.urgency_score,
            "Confirmed Mule": "YES" if inv_v.confirmed else "NO",
            "Confidence": f"{inv_v.confidence * 100:.0f}%",
            "Route": inv_v.route.upper()
        })
        
    st.dataframe(pd.DataFrame(queue_rows), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    # Deep Dive Inspector
    with st.container(border=True):
        # Account selection 
        account_options = [r.account_id for r in ranked]
        selected_acc = st.selectbox("Select Account ID to Inspect:", account_options)
        
        # Retrieve selected case data
        selected_state = next(fs for fs in results if fs["investigation_verdict"].account_id == selected_acc)
        inv_verdict = selected_state["investigation_verdict"]
        explainer = selected_state["explainer_output"]
        reviewer = selected_state["reviewer_verdict"]

        # Account header
        st.markdown(
            f"""
            <div class="case-header">
                <div class="case-eyebrow">ACCOUNT</div>
                <div class="case-account">{selected_acc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="case-divider"></div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.markdown(
                f"""
                <div class="case-info-card">
                    <div class="case-info-label">Urgency Score</div>
                    <div class="case-info-value">
                        {selected_state["urgency_score"]:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.markdown(
                '<div class="case-section-title">Gathered Evidence</div>',
                unsafe_allow_html=True,
            )

            for item in inv_verdict.evidence:
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-dot"></div>
                        <div class="evidence-text">{item}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
            with st.expander("Reviewer Gate Verdict"):
                st.write(f"**Passed Gate:** {reviewer.passed}")
                st.write(f"**Action:** {reviewer.action}")
                if reviewer.issues:
                    st.write(f"**Issues:** {', '.join(reviewer.issues)}")

        with col_b:
            st.markdown(
            '<div class="case-section-title">AI Investigator Narrative</div>',
            unsafe_allow_html=True,
            )
            st.markdown(f'<div class="narrative-box">{explainer.narrative}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Full Reasoning Trail"):
                st.write(inv_verdict.reasoning_trail)

else:
    empty_state = """
    <div class="empty-state">
    <div class="empty-state-title">No Active Batch Execution Results</div>
    <div class="empty-state-description">Upload a CSV dataset or use the default sample data, then click <strong>Run AML Triage Pipeline</strong> in the sidebar to begin.</div>
    </div>
    """

    st.markdown(empty_state, unsafe_allow_html=True)