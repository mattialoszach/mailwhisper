from pathlib import Path
from agent.graph import build_graph

if __name__ == "__main__":
    # Find correct path to save image to
    out_dir = Path(__file__).resolve().parent.parent / "img"
    out_file = out_dir / "graph.png"

    app = build_graph()

    with open(out_file, "wb") as f:
        f.write(app.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0))
        print("Graph saved as graph.png")