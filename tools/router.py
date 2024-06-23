from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

class Router:
    def __init__(self, llm_model, llm_format, llm_temperature):
        self.llm_model = llm_model
        self.llm_format = llm_format
        self.llm_temperature = llm_temperature

        self.prompt = PromptTemplate(
            template="""You are an expert at routing a 
            user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
            prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
            in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
            or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
            no premable or explaination. 
            
            Question to route: 
            {question}""",
            input_variables=["question"],
        )

        self.llm = ChatOllama(model=self.llm_model, format=self.llm_format, temperature=self.llm_temperature)
        self.router = self.prompt | self.llm | JsonOutputParser()
    
    def invoke(self, question):
        return self.router.invoke({"question": question})

###################
### Router 
###################

# router = Router(llm_model=local_llm, llm_format="json", llm_temperature=0)