from fastapi import FastAPI, APIRouter, Depends, Request, Form, UploadFile
from pydantic import BaseModel
from langchain_community.llms.ollama import Ollama
import logging
from auth import get_current_user
from typing import Annotated
from limiter import limiter
import base64
import uuid
from io import BytesIO

from abc import ABC
# import torch
from langchain_core.documents import Document
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain_community.document_loaders import PyPDFLoader,CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
import json
import os
from config import import_config
import tempfile
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate

logger = logging.getLogger(__name__)

logging.basicConfig(filename="logs.log",
                    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

router = APIRouter()

config = import_config()

CHROMA_PATH = 'chroma'

user_dependency = Annotated[dict, Depends(get_current_user)]

#user_dependency['username']

class Engine(ABC):
    def process(self,filename:str)->str:
        ...


class DocTrJsonEngine(Engine):
    def process(self, filename: str) -> str:
        if filename.endswith(".pdf"):
            doc= DocumentFile.from_pdf(filename)
        else:
            doc = DocumentFile.from_images([filename])
        predictor = ocr_predictor(pretrained=True)
        # if torch.cuda.is_available():
        #     predictor = predictor.to('cuda')
        result = predictor(doc)
    
        return result
    
# pdf_file: UploadFile
    
class PDFQuestionRequestNew(BaseModel):
    pdf_name: str
    prompt: str
    cntxt_key: str


class PDFQuestionRequestNew(BaseModel):
    pdf_base64: str
    pdf_name: str
    prompt: str
    cntxt_key: str

def get_context_from_key(key=''):
    if key == '' or key == None:
        return ''

    context = ''
    with open(config["context_file"]) as f:
        objs = json.load(f)
        # result = [x for x in objs if x["key"] == key]
        # if len(result) > 0:
        #     return result[0]['value']
        # return None       
        if key in objs:
            context = objs[key]['Context']
        else:
            context_list = [t for t in objs if objs[t]['is_default'] == True]
            if len(context_list) > 0:
                context = objs[context_list[0]]['Context']
    print(context)
    return context
    
@router.post("/ask_from_new_document")
async def ask_from_document(request: PDFQuestionRequestNew):

    doc_as_bytes = str.encode(request.pdf_base64)  # convert string to bytes
    doc = base64.b64decode(doc_as_bytes)  # decode base64string.
    act_file_name = "{id}_{name}".format(
        id=uuid.uuid4(), name=request.pdf_name)
    # location = os.path.join(config["doc_base_path"], act_file_name)
    #try:
    #    with open(location, "wb") as f:
    #        f.write(doc)
    #except Exception:
    #    logger.info("Sending 500: unable to upload file")
    #    return {"status": 500, "message": "There was an error uploading the file"}

    if request.prompt is None or request.prompt == '':
        logger.info("Sending 400: Prompt is empty")
        return {"status": 400, "message": "Prompt is empty"}

    sys_cntxt = get_context_from_key(request.cntxt_key)

    # logger.info("Context: {ct}".format(ct =sys_cntxt))
    try:
         # Write the file content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(doc)
            tmp_file_path = tmp_file.name
            print(tmp_file_path)

        try:
            # Use the temporary file path in the function
            response = get_llm_response_text(tmp_file_path, request.pdf_name, request.prompt, sys_cntxt)
        finally:
            # Make sure to delete the temporary file
            os.remove(tmp_file_path)
        # response = get_llm_response_text(file_like, pdf_file.filename,prompt, cntxt_key)

        logger.info("Request completed for new document")
        return {"status": 200, "message": "Success", "file": tmp_file_path, "response": response}
    except Exception as e:
        logger.error(e, exc_info=True)
        return {"status": 500, "message": "Internal Server Error"}


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


def create_agent_chain(llm,context):
  sys_msg_pt = SystemMessagePromptTemplate.from_template(context)
  user_pt = PromptTemplate(template="{context}\n{question}",
                                input_variables=['context', 'question'])
  user_msg_pt = HumanMessagePromptTemplate(prompt=user_pt)
  prompt = ChatPromptTemplate.from_messages([sys_msg_pt, user_msg_pt])
  qa_chain = load_qa_chain(llm, chain_type="stuff", verbose= True, prompt = prompt)
  return qa_chain
	
def get_llm_response(query, context, modeloption="llama3"):
    if modeloption == 'llama3':
        llm = Ollama(model='llama3', temperature=1)
    elif modeloption == 'chatGPT':
        llm = ChatOpenAI(model_name=config["model"])
    qa_chain = create_agent_chain(llm, context)

    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    matching_docs = db.similarity_search(query, k=5)

    print(matching_docs, "\n\n\n")
    inputs = {
        "input_documents": matching_docs,
        "question": query
    }
    return qa_chain.invoke(inputs)

def all_pages_empty(documents):
    for doc in documents:
        if doc.page_content.strip():
            return False
    return True
    
def load_chunk_persist(doc_path, file_name):
    documents = []
    # Determine file type
    file_extension = file_name.split(".").pop()

    if file_extension == 'pdf':
        # Process PDF
        
        loader = PyPDFLoader(doc_path)
        documents.extend(loader.load())
        if all_pages_empty(documents):
            documents= []
            loader = DocTrJsonEngine()
            data = loader.process(doc_path)
            data = data.export()
            
            # Converting to appropriate format
            pages = {}
            for k in range(len(data['pages'])):
                page = ""
                for i in range(len(data['pages'][k]["blocks"])):
                    for j in data['pages'][k]["blocks"][i]['lines'][0]['words']:
                        page = page + " " + j['value']
                
                pages.update({k: page})
            
            for page in pages:
                p_content = pages[page]
                m_data = {
                    'source': doc_path,
                    'page': page
                }
                d = Document(page_content=p_content, metadata=m_data)
                documents.append(d)
    elif file_extension == 'csv':
        # Process CSV
        loader = CSVLoader(doc_path)
        print(loader)
        documents.extend(loader.load())
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Print documents
    print(documents,"\n\n\n")

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=7, chunk_overlap=3)
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
   
#    #chunk id = source path : page number: chunk id
#    # data/monopoly.pdf:6:2
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks

def delete_all_documents():
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    db.delete_collection()
    db.persist()
    
def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
    chunks_with_ids = calculate_chunk_ids(chunks)
    #add or update the documents

    existing_items = db.get(include=[])
    print(existing_items,"\n\n\n")
    existing_ids = set(existing_items["ids"])

    #add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks:
       if chunk.metadata['id'] not in existing_ids:
          new_chunks.append(chunk)
    
    if len(new_chunks):
       new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
       db.add_documents(new_chunks, ids = new_chunks_ids)
       db.persist()
    else:
       pass

def get_llm_response_text(file_content, file_name, query,context):
    chunks = load_chunk_persist(file_content, file_name)
    add_to_chroma(chunks)
    return get_llm_response(query, context)