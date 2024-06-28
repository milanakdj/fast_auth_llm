from abc import ABC, abstractmethod
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain_community.document_loaders import CSVLoader

class Engine(ABC):
    @abstractmethod
    def process(self, filename: str):
        pass

class DocTrJsonEngine(Engine):
    def process(self, doc_path: str):
        if doc_path.endswith(".pdf"):
            doc = DocumentFile.from_pdf(doc_path)
        else:
            doc = DocumentFile.from_images([doc_path])
        predictor = ocr_predictor(pretrained=True)
        result = predictor(doc)
        return result.export()
        
class PyPDFLoaderEngine(Engine):
    def process(self, doc_path: str):
        loader = PyPDFLoader(doc_path)
        return loader.load()    
        
class CSVLoaderEngine(Engine):
    def process(self, doc_path: str):
        loader = CSVLoader(doc_path)
        return loader.load()
        
