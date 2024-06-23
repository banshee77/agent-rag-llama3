from langchain.globals import set_verbose, set_debug

from graph.agent_generate import generate
from graph.agent_grade_documents import grade_documents
from graph.agent_retrieve import retrieve
from graph.agent_state import AgentState
from graph.web_search import web_search
from langgraph.graph import END, StateGraph
from graph.conditional_edge.route_question import route_question
from graph.conditional_edge.decide_to_generate import decide_to_generate
from graph.conditional_edge.grade_generation_v_documents_and_question import grade_generation_v_documents_and_question 

from pprint import pprint

from qdrant.qdrant_retriever import QdrantRetriever

set_debug(True)
set_verbose(True)

workflow = StateGraph(AgentState)
workflow.add_node("websearch", web_search) 
workflow.add_node("retrieve", retrieve) 
workflow.add_node("grade_documents", grade_documents) 
workflow.add_node("generate", generate) 

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

app = workflow.compile()

# inputs = {"question": "What are the types of agent memory?"}
inputs = {"question": "Who are the Bears expected to draft first in the NFL draft?"}
# inputs = {"question": "Did Emmanuel Macron visit Germany recently?"}
# inputs = {"question": "Who is President of Poland?"}

for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])