from typing import TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph, END

from src.schemas import DetectorVerdict, InvestigationVerdict, ExplainerOutput, OutputReviewerVerdict
from src.detector import score_transaction, route as detector_route
from src.investigation_agent import investigate_account
from src.explainer_agent import explain
from src.output_reviewer_agent import review_output
from src.tools import check_velocity, check_linked_accounts


class GraphState(TypedDict):
    transaction: dict
    recipient_id: str
    detector_verdict: Optional[DetectorVerdict]
    investigation_verdict: Optional[InvestigationVerdict]
    urgency_score: Optional[float]
    explainer_output: Optional[ExplainerOutput]
    reviewer_verdict: Optional[OutputReviewerVerdict]
    rewrite_count: int


def build_graph(df: pd.DataFrame, flagged_ids: set[str], investigation_agent):

    def detector_node(state):
        row = pd.Series(state["transaction"])
        score = score_transaction(row)
        verdict = DetectorVerdict(
            account_id=row["nameOrig"], score=score,
            reason=f"type={row['type']}, amount={row['amount']:.0f}",
            route=detector_route(score),
        )
        return {"detector_verdict": verdict}

    def investigation_node(state):
        verdict = investigate_account(investigation_agent, state["recipient_id"])
        return {"investigation_verdict": verdict}

    def prioritization_node(state):
        row = pd.Series(state["transaction"])
        velocity = check_velocity(state["recipient_id"], df)
        linked = check_linked_accounts(state["recipient_id"], df, flagged_ids)
        v = state["investigation_verdict"]
        score = v.confidence * 0.4 + min(row["amount"] / 500000, 1.0) * 0.3 \
            + (0.2 if velocity.is_fast_cashout else 0.0) + min(linked.linked_count / 5, 1.0) * 0.1
        return {"urgency_score": round(min(score, 1.0), 3)}

    def explainer_node(state):
        return {"explainer_output": explain(state["investigation_verdict"])}

    def output_reviewer_node(state):
        result = review_output(state["explainer_output"], state["investigation_verdict"])
        return {"reviewer_verdict": result}

    def increment_rewrite_node(state):
        return {"rewrite_count": state["rewrite_count"] + 1}

    def detector_router(state):
        return "investigate" if state["detector_verdict"].route == "investigate" else "auto_close"

    def investigation_router(state):
        return "escalate" if state["investigation_verdict"].route == "escalate" else "close_false_alarm"

    def reviewer_router(state):
        if state["reviewer_verdict"].action == "pass":
            return "pass"
        if state["rewrite_count"] >= 1:
            return "flag_for_human_review"  # cap hit — don't loop again
        return "request_rewrite"

    graph = StateGraph(GraphState)
    graph.add_node("detector", detector_node)
    graph.add_node("investigation", investigation_node)
    graph.add_node("prioritization", prioritization_node)
    graph.add_node("explainer", explainer_node)
    graph.add_node("output_reviewer", output_reviewer_node)
    graph.add_node("increment_rewrite", increment_rewrite_node)

    graph.set_entry_point("detector")
    graph.add_conditional_edges("detector", detector_router, {"auto_close": END, "investigate": "investigation"})
    graph.add_conditional_edges("investigation", investigation_router, {"close_false_alarm": END, "escalate": "prioritization"})
    graph.add_edge("prioritization", "explainer")
    graph.add_edge("explainer", "output_reviewer")
    graph.add_conditional_edges("output_reviewer", reviewer_router, {
        "pass": END, "request_rewrite": "increment_rewrite", "flag_for_human_review": END,
    })
    graph.add_edge("increment_rewrite", "explainer")  # loop back, capped by rewrite_count

    return graph.compile()