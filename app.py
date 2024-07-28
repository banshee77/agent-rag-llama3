# https://medium.com/@zilliz_learn/local-agentic-rag-with-langgraph-and-llama-3-6c962979821f
# https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/advanced_rag/langgraph-rag-agent-local.ipynb
# https://miro.medium.com/v2/resize:fit:1400/format:webp/0*hkUfuS94m73Xm7vu.png

# the same but without a router
# https://github.com/langchain-ai/langgraph/blob/main/examples/tutorials/rag-agent-testing-local.ipynb

from langchain.globals import set_verbose, set_debug

from graph.a_agent_state import AgentState
from graph.b_route_question import route_question
from graph.c_agent_retrieve import retrieve
from graph.d_agent_grade_documents import grade_documents
from graph.e_decide_to_generate import decide_to_generate
from graph.fa_agent_generate import generate
from graph.fb_web_search import web_search
from graph.g_grade_generation_v_documents_and_question import grade_generation_v_documents_and_question 

from langgraph.graph import END, StateGraph

from pprint import pprint

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

inputs = {"question": "What are the types of agent memory?"} # QDRANT
# inputs = {"question": "Who are the Bears expected to draft first in the NFL draft?"} # WEB SERACH
# inputs = {"question": "Did Emmanuel Macron visit Germany recently?"} # WEB SERACH
# inputs = {"question": "Who is President of Poland?"} # WEB SERACH

for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])