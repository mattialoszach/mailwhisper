from langgraph.graph import StateGraph, END
from .state import DraftState
from .nodes import transcribe_node, intent_node, apply_node, decide_node

def build_graph():
    g = StateGraph(DraftState)

    g.add_node("transcribe", transcribe_node)
    g.add_node("intent", intent_node)
    g.add_node("apply", apply_node)
    g.add_node("decide", decide_node)

    g.set_entry_point("transcribe")
    g.add_edge("transcribe", "intent")
    g.add_edge("intent", "apply")
    g.add_edge("apply", "decide")

    def route(state: DraftState):
        return "finish" if state.get("done") else "loop"
    
    g.add_conditional_edges(
        "decide",
        route,
        {"loop": "transcribe", "finish": END},
    )

    return g.compile()