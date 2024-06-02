from fastapi import FastAPI, APIRouter, Depends
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

logger = logging.getLogger(__name__)

logging.basicConfig(filename="logs.log",
                    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# router = APIRouter(dependencies=[Depends(get_user)])
router = APIRouter()


class PDFQuestionRequestNew(BaseModel):
    pdf_base64: str
    pdf_name: str
    prompt: str


class PDFQuestionRequest(BaseModel):
    pdf_name: str
    prompt: str

class PromptRequest(BaseModel):
    prompt: str


#config = import_config()


@router.post("/ask_from_new_document")
async def ask_from_document(request: PDFQuestionRequestNew):
    logger.info("New document request")
    doc_as_bytes = str.encode(request.pdf_base64)  # convert string to bytes
    doc = base64.b64decode(doc_as_bytes)  # decode base64string.
    act_file_name = "{id}_{name}".format(
        id=uuid.uuid4(), name=request.pdf_name)
    location = os.path.join(config["doc_base_path"], act_file_name)
    try:
        with open(location, "wb") as f:
            f.write(doc)
    except Exception:
        logger.info("Sending 500: unable to upload file")
        return {"status": 500, "message": "There was an error uploading the file"}

    if request.prompt is None or request.prompt == '':
        logger.info("Sending 400: Prompt is empty")
        return {"status": 400, "message": "Prompt is empty"}

    try:
        response = get_llm_response(location, request.prompt)

        logger.info("Request completed for new document")
        return {"status": 200, "message": "Success", "file": act_file_name, "response": response}
    except Exception as e:
        logger.error(e, exc_info=True)
        return {"status": 500, "message": "Internal Server Error"}


@router.post("/ask_from_existing_document")
async def ask_from_document(request: PDFQuestionRequest):
    logger.info("Existing document request")
    if (request.pdf_name is None or request.pdf_name == ''):
        logger.info("Sending 400: file name is empty")
        return {"status": 400, "message": "PDF file name cannot be empty"}
    location = os.path.join(config["doc_base_path"], request.pdf_name)

    if request.prompt is None or request.prompt == '':
        logger.info("Sending 400: Prompt is empty")
        return {"status": 400, "message": "Prompt is empty"}

    if location is not None and os.path.isfile(location):
        response = get_llm_response(location, request.prompt)
        logger.info("Sending response")
        return {"status": 200, "message": "Success", "file": request.pdf_name, "response": response}
    else:
        logger.info("Sending 400: couldn't find the file to process")
        return {"status": 400, "message": "PDF file with name {name} couldn't be found in the server".format(name=request.pdf_name)}

user_dependency = Annotated[dict, Depends(get_current_user)]

@router.post("/ask")
async def ask(user:user_dependency, request: PromptRequest):

    if request.prompt is None or request.prompt == '':
        logger.info("Sending 400: Prompt is empty")
        return {"status": 400, "message": "Prompt is empty"}

    if request.prompt is not None:
        response = get_llm_response_text(request.prompt)
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