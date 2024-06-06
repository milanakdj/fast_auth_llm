from engine import Engine
from easyocr import Reader
from typing import List
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from doctr.io import DocumentFile as DocumentFileIO




os.environ['USE_TORCH'] = '1'

class EasyOCR(Engine):
    def process(self, filename: str) -> str:
        if filename.endswith(".pdf"):
            return "Error: This implementation does not support pdf format"
        # Initialize the EasyOCR reader
        reader = Reader(['en'])  # You can specify other languages as needed
        
        # Read text from the image file
        result = reader.readtext(filename)
        
        # Extract and concatenate text from the result
        text = ' '.join([entry[1] for entry in result])
        return text

    

class DocTrSyntaxEngine(Engine):
    def process(self, filename: str) -> str:
        # Load the document
        if filename.endswith(".pdf"):
            doc = DocumentFile.from_pdf(filename)
        else:
            doc = DocumentFile.from_images([filename])
        
  
        predictor = ocr_predictor(pretrained=True)
        
        if torch.cuda.is_available():
            predictor = predictor.to('cuda')
          
      
        result = predictor(doc)
        
        return str(result)
    
class DocTrJsonEngine(Engine):
    def process(self, filename: str) -> str:
        if filename.endswith(".pdf"):
            doc= DocumentFile.from_pdf(filename)
        else:
            doc = DocumentFile.from_images([filename])
        predictor = ocr_predictor(pretrained=True)
        if torch.cuda.is_available():
            predictor = predictor.to('cuda')
     
        print(predictor)
        result = predictor(doc)
       
        return result.export()