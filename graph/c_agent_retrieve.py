from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    qdrant = Qdrant.from_existing_collection(
        embedding=HuggingFaceEmbeddings(),
        collection_name="rag-agent-local",
        url="http://localhost:6333",
    )
    retriever = qdrant.as_retriever()
    
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}