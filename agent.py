import os
import requests
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

APPLICATIONS = []  # in-memory list of {"company", "role", "status"}


@tool
def add_application(company: str, role: str) -> str:
    """Add a new internship application with the status 'Applied'."""
    APPLICATIONS.append({"company": company, "role": role, "status": "Applied"})
    return f"Added: {role} at {company} (Applied)"


@tool
def list_applications() -> str:
    """List all saved internship applications with their statuses."""
    if not APPLICATIONS:
        return "No applications yet."
    return "\n".join(f"- {a['company']} | {a['role']} | {a['status']}" for a in APPLICATIONS)


@tool
def update_status(company: str, status: str) -> str:
    """Update the status (e.g. Applied, Interview, Rejected, Offer) of an application by company name."""
    for a in APPLICATIONS:
        if a["company"].lower() == company.lower():
            a["status"] = status
            return f"{a['company']} status updated to {status}"
    return f"No application found for {company}"


@tool
def get_company_info(company: str) -> str:
    """Get a short summary about a company from Wikipedia."""
    r = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{company.replace(' ', '_')}",
        headers={"User-Agent": "internship-agent/1.0"}, timeout=20)
    if r.status_code != 200:
        return f"No Wikipedia summary found for {company}"
    return r.json().get("extract", "No summary available")[:600]


@tool
def get_city_weather(city: str) -> str:
    """Get the current temperature in Celsius for a city (useful for interview day)."""
    for attempt in range(3):
        try:
            geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                               params={"name": city, "count": 1}, timeout=30).json()
            if not geo.get("results"):
                return f"City not found: {city}"
            g = geo["results"][0]
            w = requests.get("https://api.open-meteo.com/v1/forecast",
                             params={"latitude": g["latitude"], "longitude": g["longitude"],
                                     "current": "temperature_2m"}, timeout=30)
            w.raise_for_status()
            return f"{g['name']}: {w.json()['current']['temperature_2m']} °C"
        except requests.exceptions.RequestException:
            if attempt == 2:
                return "Weather service is temporarily unavailable. Please try again."


@tool
def delete_application(company: str) -> str:
    """Delete an application by company name. Requires user approval first."""
    target = next((a for a in APPLICATIONS if a["company"].lower() == company.lower()), None)
    if target is None:
        return f"No application found for {company}"
    approved = interrupt(f"Are you sure you want to delete the application to {target['company']}?")
    if approved is True:
        APPLICATIONS.remove(target)
        return f"Deleted application to {target['company']}"
    return "Action cancelled by user. Nothing was deleted."


TOOLS = [add_application, list_applications, update_status,
         get_company_info, get_city_weather, delete_application]

SYSTEM = SystemMessage(content="""You are an Internship Application Tracker assistant.
Use tools to manage applications, look up company info and check weather.
IMPORTANT: Whenever the user asks to delete, remove or cancel an application, you MUST
call the delete_application tool immediately. Never ask for confirmation in text,
the tool itself asks the user for approval.
If a tool result says the action was cancelled by the user, do NOT call the tool again.
Just tell the user the deletion was cancelled and nothing was deleted.""")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=2000)
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant(state: State):
        return {"messages": [llm_with_tools.invoke([SYSTEM] + state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=InMemorySaver())
