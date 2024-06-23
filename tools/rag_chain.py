from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

class RagChain:
    def __init__(self, llm_model, llm_temperature):
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

        self.prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise:
            Question: {question} 
            Context: {context} 
            Answer: 
            """,
            input_variables=["question", "document"],
        )

        self.llm = ChatOllama(model=self.llm_model, temperature=self.llm_temperature)
        self.rag_chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, context, question):
        return self.rag_chain.invoke({"context": context, "question": question})


###################
### Rag Chain
###################

# rag_chain = RagChain(llm_model=local_llm, llm_temperature=0)

