import streamlit as stl
from dotenv import load_dotenv
from streamlit_chat import message

from Source.retriever import get_results_from_rag

load_dotenv()

prompt = stl.header("Chat with Langchain documentation").text_input(label="Question",
                                                                    placeholder="Enter your question here")

if "historical_prompts" not in stl.session_state:
    stl.session_state["historical_prompts"] = []

if "historical_responses" not in stl.session_state:
    stl.session_state["historical_responses"] = []


def get_formated_sources(sources: set[str]) -> str:
    sources = list(sources)
    sources.sort()
    sources = [f"{idx + 1}: {word}" for (idx, word) in enumerate(sources)]
    sourceString = "\n".join(sources)
    return sourceString


if prompt:
    with stl.spinner(text="Generating the response..."):
        response = get_results_from_rag(query=prompt)
        sources = set([res.metadata["source"] for res in response["source_documents"]])
        formated_response = f"{response['result']} \n\nsources:\n{get_formated_sources(sources)}"

        stl.session_state["historical_prompts"].append(prompt)
        stl.session_state["historical_responses"].append(formated_response)

if stl.session_state["historical_responses"]:
    for (query, response) in zip(stl.session_state["historical_prompts"], stl.session_state["historical_responses"]):
        message(message=query, is_user=True)
        message(message=response)
