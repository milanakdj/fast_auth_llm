import os
import base64
import urllib.parse
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from PIL import Image
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
import shutil
from abc import ABC
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch

CHROMA_PATH = 'chroma'
DATA_PATH = 'data'

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

def get_upload_file_dialog(fu):
  location = None
  uploaded_file = fu.file_uploader("Choose a file", type="pdf")

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

def get_embedding_function():
    embedding=OllamaEmbeddings(model="nomic-embed-text")
    return embedding

def load_chunk_persist_pdf(doc_path) -> FAISS:
    documents = []
    loader = PyPDFLoader(doc_path)
    documents.extend(loader.load())
    if all_pages_empty(documents):
      loader = DocTrJsonEngine()
      data = loader.process(doc_path)
      pages = {}
      for k in range(len(data['pages'])):
          page = ""
          for i in range(len(data['pages'][k]["blocks"])):
              for j in data['pages'][k]["blocks"][i]['lines'][0]['words']:
                  page = page +" "+ j['value']
          
          pages.update({k:page})

      for page in pages:
          p_content = pages[page]
          m_data = {
            'source':doc_path,
            'page':page
          }
          d = Document(page_content= p_content, metadata =m_data)
          documents.append(d)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
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
    return chunk

def clear_database():
   if os.path.exists(CHROMA_PATH):
      shutil.rmtree(CHROMA_PATH) 

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
  pdu = st.empty()


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
    fu.empty()
    #returns chunks 
    chunks = load_chunk_persist_pdf(st.session_state.location)
    add_to_chroma(chunks)
main()