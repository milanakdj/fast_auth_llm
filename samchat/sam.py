from fastapi import FastAPI, APIRouter, Depends, Request, Form
from pydantic import BaseModel
from langchain_community.llms.ollama import Ollama
import logging
from auth import get_current_user
from typing import Annotated
from limiter import limiter

logger = logging.getLogger(__name__)

logging.basicConfig(filename="logs.log",
                    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

router = APIRouter()


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


def create_agent_chain():
    llm = Ollama(model="llama3")
    return llm


def get_llm_response_text(query):
    qa_chain = create_agent_chain()
    input = {
        "question": query
    }
    answer = qa_chain.invoke(query)
    return answer