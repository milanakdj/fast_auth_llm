from fastapi import FastAPI, APIRouter, Depends, Request, Form
from pydantic import BaseModel, Field
from typing import Annotated
import logging
import base64
import uuid
import tempfile
import os

from config import import_config
from auth import get_current_user
from limiter import limiter

from context_handler import ContextHandler
from document_processor import DocumentProcessor
from chroma_db_handler import ChromaDBHandler
from llm_handler import LLMHandler

logger = logging.getLogger(__name__)

logging.basicConfig(filename="logs.log",
                    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

router = APIRouter()

config = import_config()

context_handler = ContextHandler(config["context_file"])
document_processor = DocumentProcessor()
chroma_db_handler = ChromaDBHandler(config["chroma_path"], "nomic-embed-text")
llm_handler = LLMHandler(config)

user_dependency = Annotated[dict, Depends(get_current_user)]

class PDFQuestionRequestNew(BaseModel):
    pdf_base64: str
    pdf_name: str
    prompt: str
    cntxt_key: str= Field(default=None)
    
class QuestionRequest(BaseModel):
    prompt: str
    cntxt_key: str= Field(default=None)

@router.post("/ask_from_new_document")
@limiter.limit("50/second")
async def ask_from_document(request: PDFQuestionRequestNew, user: user_dependency):
    doc_as_bytes = str.encode(request.pdf_base64)
    doc = base64.b64decode(doc_as_bytes)
    act_file_name = "{id}_{name}".format(id=uuid.uuid4(), name=request.pdf_name)

    if not request.prompt:
        logger.info("Sending 400: Prompt is empty")
        return {"status": 400, "message": "Prompt is empty"}

    sys_cntxt = context_handler.get_context_from_key(request.cntxt_key)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(doc)
            tmp_file_path = tmp_file.name

        try:
            response = get_llm_response_text(tmp_file_path, request.pdf_name, request.prompt, sys_cntxt)
        finally:
            os.remove(tmp_file_path)

        logger.info("Request completed for new document")
        return {"status": 200, "message": "Success", "file": tmp_file_path, "response": response}
    except Exception as e:
        logger.error(e, exc_info=True)
        return {"status": 500, "message": "Internal Server Error"}

@router.post("/ask")
@limiter.limit("50/second")
async def ask(request:Request, r: QuestionRequest, user: user_dependency):
    if not r.prompt:
        logger.info("Sending 400: Prompt is empty")
        return {"status": 400, "message": "Prompt is empty"}
    sys_cntxt = context_handler.get_context_from_key(r.cntxt_key)
    response = llm_handler.get_llm_response(r.prompt, sys_cntxt)
    logger.info("Sending response")
    return {"status": 200, "message": "Success", "response": response}

def get_llm_response_text(file_content, file_name, query, context):
    chunks = document_processor.load_chunk_persist(file_content, file_name)
    chroma_db_handler.add_to_chroma(file_name, chunks)
    return llm_handler.get_llm_response(query, context)
