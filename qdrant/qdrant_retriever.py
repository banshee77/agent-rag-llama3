from langchain.globals import set_verbose, set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

class QdrantRetriever:
    def __init__(self, llm_model, chunk_size=250, chunk_overlap=0):
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.qdrant = None

    def load(self, urls):
        set_debug(True)
        set_verbose(True)

        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(docs_list)

        embeddings = HuggingFaceEmbeddings()

        self.qdrant = Qdrant.from_documents(
            documents=doc_splits,
            collection_name="rag-agent-local",
            embedding=embeddings,
        )

    def my_retriever(self):
        return self.qdrant.as_retriever()

    def retriever_from_existing_collection(self):
        embeddings = HuggingFaceEmbeddings()
        self.qdrant = Qdrant.from_existing_collection(
            embedding=embeddings,
            collection_name="rag-agent-local",
            url="http://localhost:6333",
        )
        return self.qdrant.as_retriever()

