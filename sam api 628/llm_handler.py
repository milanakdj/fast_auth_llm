from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma

class LLMHandler:
    def __init__(self, config):
        self.config = config

    def create_agent_chain(self, llm, context):
        sys_msg_pt = SystemMessagePromptTemplate.from_template(context)
        user_pt = PromptTemplate(template="{context}\n{question}", input_variables=['context', 'question'])
        user_msg_pt = HumanMessagePromptTemplate(prompt=user_pt)
        prompt = ChatPromptTemplate.from_messages([sys_msg_pt, user_msg_pt])
        return load_qa_chain(llm, chain_type="stuff", verbose=True, prompt=prompt)

    def get_llm_response(self, query, context, modeloption="llama3"):
        if modeloption == 'llama3':
            llm = Ollama(model='llama3', temperature=1)
        elif modeloption == 'chatGPT':
            llm = ChatOpenAI(model_name=self.config["model"])
        
        qa_chain = self.create_agent_chain(llm, context)
        if context:
            embedding_function = OllamaEmbeddings(model="nomic-embed-text")
            db = Chroma(persist_directory=self.config["chroma_path"], embedding_function=embedding_function)
            matching_docs = db.similarity_search(query, k=5)
        else:
            matching_docs = ""
        inputs = {
            "input_documents": matching_docs,
            "question": query
        }
        return qa_chain.invoke(inputs)
