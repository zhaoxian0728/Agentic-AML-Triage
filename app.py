import streamlit as st
import pandas as pd

from src.features import engineer_features
from src.detector import score_transaction
from src.investigation_agent import build_investigation_agent
from src.prioritization_agent import assign_ranks
from src.graph import build_graph
from src.schemas import DetectorVerdict, InvestigationVerdict, ExplainerOutput, OutputReviewerVerdict

# Toggle to True to run UI tests without making live LLM API calls
DEMO_MODE = False 

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
st.sidebar.markdown("Load a dataset and run the AML triage pipeline.")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
batch_size = st.sidebar.slider("Sample Batch Size", min_value=3, max_value=20, value=5)
st.sidebar.caption(
    "Controls how many flagged cases are investigated in this run."
)

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
    progress_placeholder = st.empty()

    def render_pipeline_progress(percent, label):
        html = f"""
<div class="pipeline-progress">
    <div class="pipeline-progress-header">
        <div class="pipeline-progress-label">AML TRIAGE PIPELINE</div>
        <div class="pipeline-progress-percent">{percent}%</div>
    </div>
    <div class="pipeline-progress-track">
        <div class="pipeline-progress-fill" style="width: {percent}%;"></div>
    </div>
    <div class="pipeline-status">
        <div class="pipeline-status-dot"></div>
        <span>{label}</span>
    </div>
</div>
"""
        progress_placeholder.markdown(html, unsafe_allow_html=True)

    render_pipeline_progress(5,"Loading transaction dataset...")
    df = prepare_data(uploaded_file)

    render_pipeline_progress(25,f"Scored {len(df):,} transactions.")
    flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])

    if not DEMO_MODE:
        render_pipeline_progress(35,"Initializing investigation agents...")
        agent = build_investigation_agent(df, flagged_ids)
        graph = build_graph(df, flagged_ids, agent)

    candidates = df[df["score"] >= 0.3]

    if len(candidates) == 0:
        progress_placeholder.empty()
        st.warning(
            "No transactions exceeded the risk threshold "
            "in the uploaded dataset."
        )
        st.stop()

    sample_batch = candidates.sample(
        n=min(batch_size, len(candidates)),
        random_state=1
    )

    render_pipeline_progress(40, f"Investigation queue prepared: "f"{len(sample_batch)} sample cases.")
    results = []

    for idx, (_, row) in enumerate(sample_batch.iterrows()):
        progress = 40 + int(
            ((idx + 1) / len(sample_batch)) * 60
        )

        render_pipeline_progress(
            progress,
            f"Investigating account {row['nameDest']} "
            f"({idx + 1}/{len(sample_batch)})..."
        )
    
        state = {
            "transaction": row.to_dict(),
            "recipient_id": row["nameDest"],
            "rewrite_count": 0,
        }

        if DEMO_MODE:
            is_escalated = idx % 2 == 0
            account_id = row["nameOrig"]

            investigation_verdict = InvestigationVerdict(
                account_id=account_id,
                confirmed=is_escalated,
                confidence=0.86 if is_escalated else 0.42,
                evidence=[
                    f"Transfer amount: ${row['amount']:,.2f}",
                    "Account activity shows patterns requiring analyst review.",
                    "Transaction was selected by the rule-based risk detector.",
                ],
                reasoning_trail=(
                    "Demo reasoning: the transaction was selected because it is a "
                    "transfer and exceeded the configured detector score threshold."
                ),
                route="escalate" if is_escalated else "close_false_alarm",
            )

            final_state = {
                "transaction": row.to_dict(),
                "detector_verdict": DetectorVerdict(
                    account_id=account_id,
                    score=float(row["score"]),
                    reason=f"type={row['type']}, amount={row['amount']:.0f}",
                    route="investigate",
                ),
                "investigation_verdict": InvestigationVerdict(
                    account_id=account_id,
                    confirmed=is_escalated,
                    confidence=0.86 if is_escalated else 0.42,
                    evidence=[
                        f"Transfer amount: ${row['amount']:,.2f}",
                        "Account activity shows patterns requiring analyst review.",
                    ],
                    reasoning_trail="Demo reasoning trail output.",
                    route="escalate" if is_escalated else "close_false_alarm",
                ),
                "explainer_output": ExplainerOutput(
                    account_id=account_id,
                    narrative="Demo narrative output for UI testing."
                ),
                "reviewer_verdict": OutputReviewerVerdict(
                    account_id=account_id,
                    passed=True,
                    issues=[],
                    action="pass",
                ),
                "urgency_score": round(
                    0.75 if is_escalated else 0.38,
                    3,
                ),
            }
        else:
            # Live LangGraph workflow call
            final_state = graph.invoke(state)

        if final_state.get("urgency_score") is not None:
            results.append(final_state)
            
        render_pipeline_progress(
            progress,
            f"Completed account {row['nameDest']}"
        )

    progress_placeholder.empty()
    st.toast(
        "Pipeline execution complete",
        icon=":material/check_circle:"
    )
    
    # Store processed results in Session State
    st.session_state["executed"] = True
    st.session_state["raw_results"] = results
    
    # Prioritization ranking calculation
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

    # Deep Dive Inspector Title & Selectbox 
    st.markdown('<div class="dash-card-title">Account Case Deep-Dive Inspector</div>', unsafe_allow_html=True)

    account_options = [r.account_id for r in ranked]
    selected_acc = st.selectbox("Select Account ID to Inspect:", account_options)
    
    # Retrieve selected case data
    selected_state = next(fs for fs in results if fs["investigation_verdict"].account_id == selected_acc)
    inv_verdict = selected_state["investigation_verdict"]
    explainer = selected_state["explainer_output"]
    reviewer = selected_state["reviewer_verdict"]

    # Inspector Card Surface
    st.markdown(
        f"""
        <div class="account-card">
            <div class="case-header">
                <div class="case-eyebrow">ACCOUNT</div>
                <div class="case-account">{selected_acc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Columns & Details
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
            '<div class="case-section-title case-section-title--first">AI Investigator Narrative</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="narrative-box">{explainer.narrative}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Full Reasoning Trail"):
            st.write(inv_verdict.reasoning_trail)

else:
    empty_state = """
<div class="empty-state">
    <div class="empty-state-main">
        <div class="empty-state-title">No Active Batch Execution</div>
        <div class="empty-state-description">
            Your investigation workspace is ready.
            Upload a transaction dataset or use the
            default sample data, then run the AML triage
            pipeline to begin.
        </div>
    </div>
    <div class="empty-state-guide">
        <div class="empty-state-guide-title">Getting Started</div>
        <div class="empty-state-step">
            <div class="empty-state-step-number">01</div>
            <div class="empty-state-step-text">Upload a transaction CSV in the Control Panel.</div>
        </div>
        <div class="empty-state-step">
            <div class="empty-state-step-number">02</div>
            <div class="empty-state-step-text">Select how many flagged cases should be investigated.</div>
        </div>
        <div class="empty-state-step">
            <div class="empty-state-step-number">03</div>
            <div class="empty-state-step-text">Run the pipeline to score, investigate, and prioritize cases.</div>
        </div>
    </div>
</div>
"""
    st.markdown(empty_state, unsafe_allow_html=True)