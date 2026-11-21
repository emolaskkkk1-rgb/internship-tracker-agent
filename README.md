# Internship Application Tracker Agent

An AI agent (LangGraph + Groq) that helps a student track internship applications:
add applications, update their status, look up company info on Wikipedia and check
the weather for an interview city (Open-Meteo). Before deleting an application the
agent pauses and asks the user for approval (human-in-the-loop with interrupt()).

## Tools
- add_application, list_applications, update_status
- delete_application (requires user approval)
- get_company_info (Wikipedia API)
- get_city_weather (Open-Meteo API)

## Setup
Install the libraries:

    pip install -r requirements.txt

Set your Groq API key (never commit a real key):

    export GROQ_API_KEY=""

## Run
- CLI: python main.py
- Streamlit: streamlit run app.py
- Notebook: open the .ipynb file in Google Colab (add GROQ_API_KEY to Colab Secrets)

## Features
StateGraph, conversation memory (InMemorySaver), approval flow (approve / deny),
5+ test queries and an evaluation table in the notebook.

## Limitations
Applications are stored in memory and reset when the app restarts. The LLM may
occasionally skip a tool call for vague phrasing (see the evaluation table).
