import os
from typing import List
import base64
import urllib.parse
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from PIL import Image
#from langchain_community.llms import ollama
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader

import asyncio
import json
from abc import ABC
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = 'chroma'

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
        if torch.cuda.is_available():
            predictor = predictor.to('cuda')
        result = predictor(doc)
    
        return result.export()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# os.environ['USE_TORCH'] = '1'

im = Image.open("images\\ai-icon-small.png")
st.set_page_config(page_title="SAM-llama3",page_icon=im, layout="wide",initial_sidebar_state="collapsed")

def get_base64_of_bin_file(bin_file):
  with open(bin_file, 'rb') as f:
      data = f.read()
  return base64.b64encode(data).decode()

def handle_pdf_select():
   if st.session_state.kind_of_pdf:
    st.session_state.pdf = st.session_state.kind_of_pdf


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden !important}
footer {visibility: hidden !important;}
.stDeployButton {visibility: hidden !important;}
.block-container {
  padding-top: 2rem;
  padding-bottom: 0rem;
  padding-left: 2rem;
  padding-right: 2rem;
}
[data-testid="stHorizontalBlock"] {
  gap: 0rem;
}
[data-testid="column"]:nth-of-type(2) {
  align-content: center;
  min-width: 42px;
  min-height: 50px;
}
[data-testid="column"]:nth-of-type(2) [data-testid=stVerticalBlock] {
  text-align: center;
}
[data-testid="column"]:nth-of-type(1) [data-testid=stVerticalBlockBorderWrapper] {
  padding-top: 0rem;
  padding-left: 0rem;
  padding-right: 0rem;
}
#text_area_1 {
  background: black;
  border: 1px solid #233138;
  width: 100%;
  min-height: 528px;
  border-radius: 5px;
  color: #fff;
  padding: 15px;
}
.st-c1, .st-c2, .st-c3, .st-c4 {
  border-color: rgba(49, 51, 63, 0.2)!important;
},
.st-b1 {
  background: transparent !important;
}

.dropdown-container{
display:flex;
    margin-left:400px;
    text-align:right;
justify-content: flex-end;    
}

.stSelectbox > div{ 
    width:200px;
    float:left;
    }

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

bin_str = get_base64_of_bin_file("images\\ai-ask.png")
page_bg_img = '''
  <style>
    [data-testid="column"]:nth-of-type(1) button,
    [data-testid="column"]:nth-of-type(1) button:focus {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
      width: 176px;
      height: 87px;
      border: 0px;
      background-color: transparent !important;
      transition: none !important;
      border: none !important;
      position: absolute;
    }
    [data-testid="column"] [data-testid="stButton"]
    {
      position: relative;
      bottom: 115px;
      right: 154px;
      text-align: right;
    }
  </style>
  ''' % bin_str
    
st.markdown(page_bg_img, unsafe_allow_html=True)

# with open("config.json") as user_file:
#   file_contents = user_file.read()
# config = json.loads(file_contents)

# model_id = config["model"]

def decode_file_path(encoded_file_path):

  # URL decode the Base64 string
  decoded_base64_string = urllib.parse.unquote(encoded_file_path)

  # Decode Base64 string to get the file path
  decoded_file_path_bytes = base64.b64decode(decoded_base64_string)

  return decoded_file_path_bytes.decode('utf-8')


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

def click_button():
    st.session_state.clicked = True


def all_pages_empty(documents):
  for doc in documents:
      if doc.page_content.strip():  
          return False
  return True




with open("config.json") as user_file:
  file_contents = user_file.read()
  config = json.loads(file_contents)


if 'filename' not in st.session_state:
    st.session_state.filename = config['context_path']


def create_agent_chain(llm):
  file = json.loads(Path(st.session_state.filename).read_text())
  context = ""
  for f in file:
      if file[f]['is_default'] == True:
        context = file[f]['Context']
  sys_msg_pt = SystemMessagePromptTemplate.from_template(context)
  user_pt = PromptTemplate(template="{context}\n{question}",
                                input_variables=['context', 'question'])
  user_msg_pt = HumanMessagePromptTemplate(prompt=user_pt)
  prompt = ChatPromptTemplate.from_messages([sys_msg_pt, user_msg_pt])
  qa_chain = load_qa_chain(llm, chain_type="stuff", verbose= True, prompt = prompt)
  return qa_chain


def get_embedding_function():
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    return embedding    
def transform_data(initial_data):
    transformed_data = []
    for doc_tuple in initial_data:
        document, _ = doc_tuple
        # Remove the 'id' key from the metadata
        new_metadata = {k: v for k, v in document.metadata.items() if k != 'id'}
        # Create new Document with the updated metadata
        new_document = Document(page_content=document.page_content, metadata=new_metadata)
        transformed_data.append(new_document)
    return transformed_data

def get_llm_response(location, query:str, modeloption):
    llm = None

    if modeloption == 'llama3':
        llm = Ollama(model='llama3')
    elif modeloption == 'chatGPT':
        pass
    qa_chain = create_agent_chain(llm)

    # matching_docs = vectordb.similarity_search(query)

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    matching_docs = db.similarity_search_with_score(query, k=5)

    results = transform_data(matching_docs)
    print(results, "\n\n\n")
    inputs = {
        "input_documents": results,
        "question": query
    }
    return inputs, qa_chain


def main():
    
    with st.container():
        st.markdown('<div class= "floating">', unsafe_allow_html = True)
        modeloption = st.selectbox("Choose LLM Model", ['chatGPT','llama3'], index =1)
        st.markdown('</div>', unsafe_allow_html=True)  
        
        
    placeholder = None

    filename = ''

    c1,c2,c3 = st.columns([19,1.1,19], gap="small")

    with c1:
        prompt = st.text_area("",placeholder="Ask me anything!", label_visibility='collapsed')
        b = Image.open("images\\ai-ask.png")
        st.button("", on_click=click_button, key="ask_button")

    with c2:
        pass

    with c3.container(border=1, height=528):
        placeholder = st.empty()
        placeholder.write("SAM is waiting..")

    if prompt:
        inputs, qa_chain = get_llm_response("",prompt, modeloption)
        async def print_chain():
            full_response = ""
            async for event in qa_chain.astream_events(inputs, version = 'v1'):
                kind = event["event"]
                if kind == "on_llm_stream":
                    full_response += event['data']['chunk']
                    placeholder.markdown(full_response,unsafe_allow_html=True)

        asyncio.run(print_chain())

        st.session_state.clicked = False
main()