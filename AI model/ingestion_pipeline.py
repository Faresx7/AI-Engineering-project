import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader, DirectoryLoader    # for reading files
from langchain_text_splitters import RecursiveCharacterTextSplitter      # for chunking
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma     # for vector DB
import shutil
from pathlib import Path


class IngestionPipeline:
    def __init__(self,
                 docs_dir = Path(__file__).resolve().parent / "docs",
                 db_dir = Path(__file__).resolve().parent / "db" / "chroma_db",
                 embedding_model_name = "all-MiniLM-L6-v2",
                 chunk_size = 650,
                 chunk_overlap = 70,
                 ):
        
        self.doc_dir = docs_dir
        self.db_dir = db_dir
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def _load_documents(self):

        if not self.doc_dir.exists():
            raise FileNotFoundError(f'Path "{self.doc_dir}" does not exist')

        loader = DirectoryLoader(
            path = str(self.doc_dir),
            glob = "*.txt",     # filters files only want to read
            loader_cls = TextLoader     # define the class that responsible for reading files inside the folder
            )
        
        docs = loader.load()

        if len(docs) == 0:
            raise FileNotFoundError(f'There is no files in "{self.doc_dir}" folder')

        return docs
        

    def _chunk_documents(self,docs):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size,
                                                    chunk_overlap = self.chunk_overlap,     # helps model keep the meaning
                                                    separators=["\n", " ", ""]
                                                    )

        chunks = text_splitter.split_documents(docs)

        return chunks


    def _create_vector_store(self, chunks):
        """
        Embeds text chunks using HuggingFace model and stores them in Chroma DB.
        Model max context capacity: 256 tokens (~600-700 chars).
        """
        
        if self.db_dir.exists():
            shutil.rmtree(self.db_dir)

        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        vector_store = Chroma.from_documents(
            documents= chunks,
            embedding = embedding_model,
            persist_directory = str(self.db_dir),
            collection_metadata= {"hnsw:space":"cosine"}
        )
        
        return vector_store


    def get_doc_info(self,doc):
            '''
            return additional informations about file
            '''
            return f'Letters: {len(doc.page_content)}, Words: {len(doc.page_content.split())}, MetaData: {doc.metadata}'


    def run(self, verbose = True):

        if verbose:
            print("🚀 [1/3] Loading documents...")
        docs = self._load_documents()

        if verbose:
            print(f"✂️ [2/3] Chunking {len(docs)} document(s)...")
        chunks = self._chunk_documents(docs)

        if verbose:
            print(f"💾 [3/3] Embedding & saving {len(chunks)} chunks to Chroma DB...")
        vector_store = self._create_vector_store(chunks)

        if verbose:
            print("✅ Ingestion Pipeline completed successfully!")

        return vector_store





if __name__ == "__main__":
    pipeline = IngestionPipeline()

    vdb = pipeline.run()

