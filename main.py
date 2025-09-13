from agent.state import initial_state
from agent.graph import build_graph
from ui.app import main

# Only for Running with CLI/Debug
def print_state(state):
    print("\n—— Current Draft ———————————————————")
    if state.get("to"):      print("To:     ", ", ".join(state["to"]))
    if state.get("cc"):      print("Cc:     ", ", ".join(state["cc"]))
    if state.get("subject"): print("Subject:", state["subject"])
    print("Tone:   ", state.get("tone", "neutral"))
    print("Body:\n" + (state.get("body", "(empty)") or "(empty)"))
    print("—————————————————————————————————————\n")

def chat_session():
    app = build_graph()

    print("\n📧 LangGraph Mail-Drafter (MVP) – structured output\n")
    
    # To display intermediate states, we use streaming
    state = initial_state()
    for event in app.stream(state, config={"recursion_limit": 300}):
        node, patch = next(iter(event.items()))
        if isinstance(patch, dict):
            state.update(patch)
        if node == "apply":
            print_state(state)
    
    if state is not None:
        print("\n—————————————————————————————————————\n")
        print("✅ Finished")
        print_state(state)
        print("\n—————————————————————————————————————")

if __name__ == "__main__":
    main()