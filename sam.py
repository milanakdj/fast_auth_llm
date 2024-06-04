from fastapi import FastAPI, APIRouter, Depends, Request, Form
# from config import import_config
from pydantic import BaseModel
# from auth import get_user
import base64
import uuid
import os
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
import logging
from auth import get_current_user
from typing import Annotated
from limiter import limiter

router = APIRouter()


logger = logging.getLogger(__name__)

logging.basicConfig(filename="logs.log",
                    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# router = APIRouter(dependencies=[Depends(get_user)])


class PromptRequest(BaseModel):
    prompt: str

user_dependency = Annotated[dict, Depends(get_current_user)]

@router.post("/ask")
@limiter.limit("50/second")
async def ask(request: Request, user:user_dependency,  prompt: Annotated[str, Form()]):

    if prompt is None or prompt == '':
        logger.info("Sending 400: Prompt is empty")
        return {"status": 400, "message": "Prompt is empty"}

    if prompt is not None:
        response = get_llm_response_text(prompt)
        logger.info("Sending response")
        return {"status": 200, "message": "Success","response": response}
    else:
        logger.info("Sending 400: couldn't find the file to process")
        return {"status": 400, "message": "error cannot reply"}



def load_chunk_persist_pdf(doc_path) -> FAISS:
    documents = []
    loader = PyPDFLoader(doc_path)
    documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)

    vectordb = FAISS.from_documents(
        documents=chunked_documents,
        embedding=OllamaEmbeddings(model="nomic-embed-text")
    )

    return vectordb


def create_agent_chain():
    llm = Ollama(model="llama3")
    return llm


def get_llm_response(location, query):
    vectordb = load_chunk_persist_pdf(location)
    qa_chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    inputs = {
        "input_documents": matching_docs,
        "question": query
    }
    answer = qa_chain.invoke(inputs)
    return answer["output_text"]

def get_llm_response_text(query):
    qa_chain = create_agent_chain()
    input = {
        "question": query
    }
    answer = qa_chain.invoke(query)
    return answer