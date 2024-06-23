from qdrant_retriever import QdrantRetriever

local_llm = "llama3"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

qdrant_retriever = QdrantRetriever(local_llm)
qdrant_retriever.load(urls)