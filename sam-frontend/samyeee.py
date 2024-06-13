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
from streamlit_js_eval import streamlit_js_eval

import asyncio
import json
from abc import ABC
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from pathlib import Path

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
st.set_page_config(page_title="SAM-llama3",page_icon=im, layout="wide")

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

if 'json_val' not in st.session_state:
   st.session_state.json_val = ""

def edit_json():
  file_path =Path(st.session_state.filename)
  print(st.session_state.text_value,"is the updateeeeeeeeeeeeeeeeeee")
  if st.session_state.json_val != st.session_state.text_value:
    st.session_state.json_val = st.session_state.text_value
    file_path.write_text(st.session_state.json_val)
    
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


def all_pages_empty(documents):
  for doc in documents:
      if doc.page_content.strip():  
          return False
  return True


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
    chunked_documents = text_splitter.split_documents(documents)

    vectordb = FAISS.from_documents(
        documents=chunked_documents,
        embedding=OllamaEmbeddings(model="nomic-embed-text")
    )
    return vectordb


def get_llm_response(location, query, modeloption):
    vectordb = load_chunk_persist_pdf(location)
    llm = None

    if modeloption == 'llama3':
        llm = Ollama(model='llama3')
    elif modeloption == 'chatGPT':
        # llm = ChatOpenAI(model_name=config["model"])
        pass
    callback_manager1 = CallbackManager([StreamingStdOutCallbackHandler()])
    # qa_chain = load_qa_chain(llm, chain_type="stuff", callback_manager=callback_manager1, verbose= True)
    from langchain_core.output_parsers import StrOutputParser
    qa_chain = load_qa_chain(llm, chain_type="stuff", verbose= True)
    # qa_chain = load_qa_chain(llm, chain_type="stuff")
    matching_docs = vectordb.similarity_search(query)
    
    inputs = {
        "input_documents": matching_docs,
        "question": query
    }
    return inputs, qa_chain


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
  
  if st.session_state.filename is not None: 
    st.write('You selected `%s`' % st.session_state.filename)
    fu.empty()
    c1,c2 = st.columns([0.7, 0.3], gap="small")
    with c1:
      c = json.loads(Path(st.session_state.filename).read_text())

      json_val = st.text_area(label='json',key='text_value', value=json.dumps(c, indent=4))
      print("\n\n\n",json_val,"\n\n\n")

      b = Image.open("images\\ai-ask.png")
      fname = str(st.session_state.filename)
      st.markdown("""
          <script>
          function reloadPage() {
              window.location.reload(true);
          }
          </script>
      """, unsafe_allow_html=True)
    with c2:
      st.markdown(
      """
        <style>

        .stButton>button {
          background-color: #4F8BF9;
          color: black;
          border-radius: 50%;
          height: 5em;
          width: 5em;
      }

      .stTextInput>div>div>input {
          color: #4F8BF9;
      }
        </style>
        """, unsafe_allow_html=True)
      st.button("Submit", on_click=edit_json, key="ask_button2")


    
  if st.session_state.location is not None and os.path.isfile(st.session_state.location):
    # loaders = None
    # loaders = PyPDFLoader(st.session_state.location)
    # #documents = loaders.load()
  
    # #llm = OpenAI(model_name = model_id)

    # # chain = load_qa_chain(llm,verbose=True)

    # # Create a vector representation of this document loaded
    # index = None
    # embeddings = OpenAIEmbeddings()
    # index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loaders])
    fu.empty()
    with st.container():
        st.markdown('<div class= "floating">', unsafe_allow_html = True)
        modeloption = pdu.selectbox("Choose LLM Model", ['chatGPT','llama3'], index =1)
        st.markdown('</div>', unsafe_allow_html=True)  
        
    placeholder = None

    filename = ''

    if 'name' in st.query_params and st.query_params["name"]!= '':
      filename = st.query_params["name"]
    else:
      filename = os.path.basename(st.session_state.location)

    # st.title('Ask SAM')
    if filename!='':
      st.write(f"Document: :red[{filename}]")

    c1,c2,c3 = st.columns([19,1.1,19], gap="small")
    
    
    with c1:
      prompt = st.text_area("",placeholder="Ask me anything!", label_visibility='collapsed')
      b = Image.open("images\\ai-ask.png")
      st.button("", on_click=click_button, key="ask_button")

    with c2:
      # f = open("images\\center-img.svg","r")
      # lines = f.readlines()
      # line_string=''.join(lines)
      # render_svg(line_string)
      pass

    with c3.container(border=1, height=528):
      placeholder = st.empty()
      placeholder.write("SAM is waiting..")

    if st.session_state.clicked and prompt:
        inputs, qa_chain = get_llm_response(st.session_state.location, prompt, modeloption)
        async def print_chain():
          full_response = ""
          async for event in qa_chain.astream_events(inputs, version = 'v1'):
            kind = event["event"]
            if kind == "on_llm_stream":
              full_response += event['data']['chunk']
              placeholder.markdown(full_response,unsafe_allow_html=True)

        # Running the asyncio event loop
        asyncio.run(print_chain())

       
        # input2 = query + " based on "+ matching_docs[0]['page_content']
        # async def print_chain():
        #     result = "\n\n".join(doc.page_content for doc in inputs['input_documents'])
        #     llm = Ollama(model='llama3')
        #     input2 = inputs['question'] + " based on " + result
        #     full_response = ""
        #     async for line in llm._astream(input2):
        #       full_response += line.text
        #       placeholder.markdown(full_response,unsafe_allow_html=True)
        # # Running the asyncio event loop
        # asyncio.run(print_chain())
        
        # async def print_chain():
        #     full_response = ""
        #     async for word in qa_chain.astream(inputs):
        #       print(word)
        #       full_response += word['output_text']
        #       placeholder.markdown(full_response,unsafe_allow_html=True)

        # # Running the asyncio event loop
        # asyncio.run(print_chain())

        # def print_chain():
        #     full_response = ""
        #     for word in qa_chain.stream(inputs):
        #       print(word['output_text'], end = "", flush = True)
        #       full_response += word['output_text']
        #       placeholder.markdown(full_response,unsafe_allow_html=True)

        # print_chain()

      #  placeholder.write("<b>" + prompt + "</b>", unsafe_allow_html=True)
       
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