from engine import DocTrJsonEngine, CSVLoaderEngine, PyPDFLoaderEngine, Engine

class EngineFactory:
    @staticmethod
    def get_text_percentage(file_name: str) -> float:
        total_page_area = 0.0
        total_text_area = 0.0
        doc = fitz.open(file_name)

        for page_num, page in enumerate(doc):
            # Calculate the area of the page
            total_page_area += page.rect.width * page.rect.height
            
            text_area = 0.0
            for b in page.get_text("blocks"):
                r = fitz.Rect(b[:4])  # Rectangle where block text appears
                text_area += r.width * r.height
            total_text_area += text_area
        
        doc.close()
        
        if total_page_area == 0:
            return False
        
        if (total_text_area / total_page_area) >.25:
            return True
        else:
            return False
            
    @staticmethod
    def create_engine(file_extension: str, doc_path:str) -> Engine:
        if file_extension == 'pdf':
            if self.get_text_percentage(doc_path):
                return PyPDFLoaderEngine()
            else:
                return DocTrJsonEngine()
        elif file_extension == 'csv':
            return CSVLoaderEngine()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
