from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

class DocumentProcessor:

    #@staticmethod
    #def all_pages_empty(documents):
    #    return all(not doc.page_content.strip() for doc in documents)

    def load_chunk_persist(self, doc_path, file_name):
        file_extension = file_name.split(".").pop()
        engine = create_engine(file_extension, doc_path)

        if file_extension == 'pdf':
            documents = engine.process(doc_path)
            text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100, length_function=len)
        elif file_extension == 'csv':
            documents = engine.process(doc_path)
            text_splitter = CharacterTextSplitter(separator = ",",chunk_size=3000, chunk_overlap=0, length_function=len)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return text_splitter.split_documents(documents)

    @staticmethod
    def calculate_chunk_ids(file_name,chunks):
        last_page_id = None
        current_chunk_index = 0
        for chunk in chunks:
            source = file_name
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
        return chunks
