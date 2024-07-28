
from tools.rag_chain import RagChain

local_llm = "llama3"

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    rag_chain = RagChain(llm_model=local_llm, llm_temperature=0)

    # RAG generation
    generation = rag_chain.invoke(documents,question)
    return {"documents": documents, "question": question, "generation": generation}