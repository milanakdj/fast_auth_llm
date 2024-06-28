from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

class ChromaDBHandler:
    def __init__(self, chroma_path: str, model: str):
        self.chroma_path = chroma_path
        self.embedding_function = OllamaEmbeddings(model=model)

    def delete_all_documents(self):
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        db.delete_collection()
        db.persist()

    def add_to_chroma(self,file_name, chunks):
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        chunks_with_ids = DocumentProcessor.calculate_chunk_ids(file_name,chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])

        new_chunks = [chunk for chunk in chunks if chunk.metadata['id'] not in existing_ids]

        if new_chunks:
            new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunks_ids)
            db.persist()
