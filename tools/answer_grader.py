from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

class AnswerGrader:
    def __init__(self, llm_model="llama3", llm_format="json", llm_temperature=0):
        self.llm_model = llm_model
        self.llm_format = llm_format
        self.llm_temperature = llm_temperature

        self.prompt = PromptTemplate(
            template="""You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            
            Here is the answer:
            {generation} 

            Here is the question: {question}
            """,
            input_variables=["generation", "question"],
        )

        self.llm = ChatOllama(model=self.llm_model, format=self.llm_format, temperature=self.llm_temperature)
        self.answer_grader = self.prompt | self.llm | JsonOutputParser()

    def invoke(self, question, generation):
        return self.answer_grader.invoke({"question": question,"generation": generation})

#########################
### Answer Grader
#########################

# answer_grader = AnswerGrader(llm_model=local_llm, llm_format="json", llm_temperature=0)
