
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

class HallucinationGrader:
    """
    Hallucination Grader

    Assesses whether an answer is grounded in / supported by a set of facts.
    """

    def __init__(self, llm_model="llama3", llm_format="json", llm_temperature=0):
        self.llm_model = llm_model
        self.llm_format = llm_format
        self.llm_temperature = llm_temperature

        self.prompt = PromptTemplate(
            template="""You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation.
            
            Here are the facts:
            {documents} 

            Here is the answer: 
            {generation}
            """,
            input_variables=["generation", "documents"],
        )

        self.llm = ChatOllama(model=self.llm_model, format=self.llm_format, temperature=self.llm_temperature)
        self.hallucination_grader = self.prompt | self.llm | JsonOutputParser()

    def invoke(self, documents, generation):
        return self.hallucination_grader.invoke({"documents": documents, "generation": generation})


#########################
### Hallucination Grader
#########################

# hallucination_grader = HallucinationGrader(llm_model=local_llm, llm_format="json", llm_temperature=0)
