import os
from typing import List
import base64
import urllib.parse
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader,CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from PIL import Image
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from streamlit_js_eval import streamlit_js_eval
from pathlib import Path
from langchain.vectorstores.chroma import Chroma

import json
import asyncio
import ssl
from abc import ABC
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
import os
import shutil

ssl._create_default_https_context = ssl._create_unverified_context

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['USE_TORCH'] = 'YES'
os.environ['USE_TF'] = 'NO'

if "context" not in st.session_state:
    st.session_state['context'] = ""

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
    
        return result

im = Image.open("images\\ai-icon-small.png")
st.set_page_config(page_title="SAM-llama3",page_icon=im, layout="wide", initial_sidebar_state = 'collapsed')

def get_base64_of_bin_file(bin_file):
  with open(bin_file, 'rb') as f:
      data = f.read()
  return base64.b64encode(data).decode()
  
def handle_pdf_select():
   if st.session_state.kind_of_pdf:
    st.session_state['loader'] = st.session_state.kind_of_pdf


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
  background: transparent;
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
    width:300px;}
</style>
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

with open("config.json") as user_file:
  file_contents = user_file.read()
config = json.loads(file_contents)

model_id = config["model"]

CHROMA_PATH = 'chroma'
DATA_PATH = 'data'

if 'filename' not in st.session_state:
    st.session_state.filename = config['context_path']
  

def decode_file_path(encoded_file_path):

  # URL decode the Base64 string
  decoded_base64_string = urllib.parse.unquote(encoded_file_path)

  # Decode Base64 string to get the file path
  decoded_file_path_bytes = base64.b64decode(decoded_base64_string)

  return decoded_file_path_bytes.decode('utf-8')

def get_upload_file_dialog(fu):
  location = None
  uploaded_file = fu.file_uploader("Choose a file", type=["pdf", "csv"])

  if uploaded_file is not None:
    # save the file temporarily
    location = os.path.join('tmp', uploaded_file.name)

    with open(location, "wb") as file:
       file.write(uploaded_file.getvalue())

  return location

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
    

def load_chunk_persist(doc_path):
    documents = []

    # Determine file type
    file_extension = os.path.splitext(doc_path)[1].lower()

    if file_extension == '.pdf':
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
    elif file_extension == '.csv':
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


def create_agent_chain(llm):
  file = json.loads(Path(st.session_state.filename).read_text())
  sys_msg_pt = SystemMessagePromptTemplate.from_template(st.session_state["context"])
  user_pt = PromptTemplate(template="{context}\n{question}",
                                input_variables=['context', 'question'])
  user_msg_pt = HumanMessagePromptTemplate(prompt=user_pt)
  prompt = ChatPromptTemplate.from_messages([sys_msg_pt, user_msg_pt])
  qa_chain = load_qa_chain(llm, chain_type="stuff", verbose= True, prompt = prompt)
  return qa_chain
	
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
    
def get_llm_response(location, query, modeloption):
    
    if modeloption == 'llama3':
        llm = Ollama(model='llama3', temperature=1)
    elif modeloption == 'chatGPT':
        llm = ChatOpenAI(model_name=config["model"])
    qa_chain = create_agent_chain(llm)

    # matching_docs = vectordb.similarity_search(query)

    embedding_function = OpenAIEmbeddings()
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
    return chunk

def delete_all_documents():
    # Initialize the Chroma database connection
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=OpenAIEmbeddings()
    )

    # Fetch all existing documents
    #existing_items = db.get(include=[])
    #existing_ids = set(existing_items["ids"])

    # Delete all documents
    #for document_id in existing_ids:
        #print('deleting', document_id)
        #db.delete(document_id)

    # Persist the changes
    db.delete_collection()
    db.persist()
    
if 'g' not in st.session_state:
    st.session_state['g'] = True
    delete_all_documents()

def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

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
       st.write(f"adding new documents: {len(new_chunks)}")
       new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
       db.add_documents(new_chunks, ids = new_chunks_ids)
       db.persist()
    else:
       st.write("no new documents added")
    

def main():

  if 'clicked' not in st.session_state:
    st.session_state.clicked = False

  if 'doc' not in st.session_state:
    st.session_state.location =''
    
  type = 0
  
  
  if 'type' in st.query_params:
    type = st.query_params['type']

  doc = None

  fu = st.empty()
  pu = st.empty()
  
  if 'doc' in st.query_params:
    doc = st.query_params["doc"]
    if doc is not None and doc != '':
      if type == '1': #doc has complete path in encoded format
        st.session_state.location  = decode_file_path(doc)
      else:
        st.session_state.location = os.path.join(config["doc_base_path"], doc)
    else:
      st.session_state.location = get_upload_file_dialog(fu)
    #st.query_params.clear()
  else:
    st.session_state.location = get_upload_file_dialog(fu)

    if st.session_state.location is not None and os.path.isfile(st.session_state.location):

        chunks = load_chunk_persist(st.session_state.location)
        add_to_chroma(chunks)
    # loaders = None
    # loaders = PyPDFLoader(st.session_state.location)
    # #documents = loaders.load()
  
    # #llm = OpenAI(model_name = model_id)

    # # chain = load_qa_chain(llm,verbose=True)

    # # Create a vector representation of this document loaded
    # index = None
    # embeddings = OpenAIEmbeddings()
    # index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loaders])
    #fu.empty()
    #pu.empty()
  with st.container():
    st.markdown('<div class= "dropdown-container">', unsafe_allow_html = True)
    file = json.loads(Path(st.session_state.filename).read_text())
    context_list = [f for f in file]
    i=0
    for f in file:
      if file[f]['is_default'] == True:
        break
      else:
        i=i+1
    context = st.selectbox("Choose Configuration", context_list, index =i)
    
    for f in file:
      if f == context:
        st.session_state['context'] = file[f]['Context']
    print(st.session_state['context'],"\n\n\n")
    st.markdown('</div>', unsafe_allow_html=True)
        
    with st.container():
        st.markdown('<div class= "dropdown-container">', unsafe_allow_html = True)
        modeloption = st.selectbox("Choose LLM Model", ['chatGPT','llama3'], index =0)
        st.markdown('</div>', unsafe_allow_html=True)
    placeholder = None

    filename = ''

    if 'name' in st.query_params and st.query_params["name"]!= '':
      filename = st.query_params["name"]
    else:
        pass
      #filename = os.path.basename(st.session_state.location)

    # st.title('Ask SAM')
    if filename!='':
        st.write(f"Document: :red[{filename}]")

    c1,c2,c3 = st.columns([19,1.1,19], gap="small")


  with c1:
    prompt = st.text_area("",placeholder="Ask me anything!", label_visibility='collapsed')
    b = Image.open("images\\ai-ask.png")
    st.button("", on_click=click_button, key="ask_button")

  with c2:
    f = open("images\\center-img.svg","r")
    lines = f.readlines()
    line_string=''.join(lines)
    render_svg(line_string)

  with c3.container(border=1, height=528):
    placeholder = st.empty()
    placeholder.write("SAM is waiting..")

  if st.session_state.clicked and prompt:
    inputs, qa_chain = get_llm_response(st.session_state.location,prompt,modeloption)

    #placeholder.write("<b>" + prompt + "</b>", unsafe_allow_html=True)
    async def print_chain():
      full_response = ""
      async for event in qa_chain.astream_events(inputs, version = 'v1'):
        kind = event["event"]
        if modeloption == 'llama3':
          if kind == "on_llm_stream":
            full_response += event['data']['chunk']
            placeholder.markdown(full_response,unsafe_allow_html=True)
        else:
            if kind == 'on_chat_model_stream':
              full_response += event['data']['chunk'].content
              placeholder.markdown(full_response,unsafe_allow_html=True)

    # Running the asyncio event loop
    asyncio.run(print_chain())
    st.session_state.clicked = False
    # Display the current response. No chat history is maintained
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM
    # 
    #Replacing CHATGPT by Llama3 
    # if modeloption == 'llama3':
    # response = index.query(llm=Ollama(model='llama3'), question = prompt, chain_type = 'stuff')
    # elif modeloption =='chatGPT':
    # response = index.query(llm=OpenAI(model_name=model_id, temperature=0.9), question = prompt, chain_type = 'stuff')
    #response = chain.run(input_documents=documents, question=prompt)
    # Write the results from the LLM to the UI

main()