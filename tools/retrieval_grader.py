from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

class RetrievalGrader:
    def __init__(self, llm_model, llm_format, llm_temperature):
        self.llm_model = llm_model
        self.llm_format = llm_format
        self.llm_temperature = llm_temperature

        self.prompt = PromptTemplate(
            template="""You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
            
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
             
            Here is the retrieved document: 
            {document}
            
            Here is the user question: 
            {question}
            """,
            input_variables=["question", "document"],
        )

        self.llm = ChatOllama(model=self.llm_model, format=self.llm_format, temperature=self.llm_temperature)
        self.retrieval_grader = self.prompt | self.llm | JsonOutputParser()

    def invoke(self, question, document):
        return self.retrieval_grader.invoke({"question": question, "document": document})


###################
### Retrieval Grader 
###################

# retrieval_grader = RetrievalGrader(llm_model=local_llm, llm_format="json", llm_temperature=0)

