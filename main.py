import os
import getpass
from langgraph.types import Command

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter GROQ_API_KEY: ").strip()

from agent import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": "cli-1"}}

print("Internship Tracker Agent. Type 'exit' to quit.")
while True:
    text = input("\nYou: ").strip()
    if text.lower() in ("exit", "quit"):
        break
    result = graph.invoke({"messages": [("user", text)]}, config)
    while "__interrupt__" in result:
        question = result["__interrupt__"][0].value
        answer = input(f"\n[CONFIRM] {question} (yes/no): ").strip().lower()
        result = graph.invoke(Command(resume=answer in ("yes", "y")), config)
    print("\nAgent:", result["messages"][-1].content)
